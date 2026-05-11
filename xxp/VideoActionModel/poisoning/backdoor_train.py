import torch
import torch.nn.functional as F


def _as_btchw(video):
    """Normalize supported layouts to [B, T, C, H, W]."""
    if video.dim() != 5:
        raise ValueError(f"expected a 5D video tensor, got shape {tuple(video.shape)}")

    # Existing dataset returns [B, C, T, H, W].
    if video.shape[1] in (1, 3):
        return video.permute(0, 2, 1, 3, 4)

    # Some video models use [B, T, C, H, W].
    if video.shape[2] in (1, 3):
        return video

    raise ValueError(
        "could not infer video layout; expected [B,C,T,H,W] or [B,T,C,H,W], "
        f"got {tuple(video.shape)}"
    )


def _as_bthw(mask):
    """Normalize optional person masks to [B, T, H, W]."""
    if mask.dim() == 5 and mask.shape[1] == 1:
        return mask[:, 0]
    if mask.dim() == 5 and mask.shape[2] == 1:
        return mask[:, :, 0]
    if mask.dim() == 4:
        return mask
    raise ValueError(f"expected mask shape [B,T,H,W], [B,1,T,H,W], or [B,T,1,H,W], got {tuple(mask.shape)}")


def _luminance(video_btchw):
    weights = video_btchw.new_tensor([0.299, 0.587, 0.114]).view(1, 1, 3, 1, 1)
    if video_btchw.shape[2] == 1:
        return video_btchw[:, :, 0]
    return (video_btchw * weights).sum(dim=2)


def _gradient_energy(gray_bthw):
    dx = gray_bthw[..., :, 1:] - gray_bthw[..., :, :-1]
    dy = gray_bthw[..., 1:, :] - gray_bthw[..., :-1, :]
    dx = F.pad(dx.abs(), (0, 1, 0, 0))
    dy = F.pad(dy.abs(), (0, 0, 0, 1))
    return dx + dy


def _soft_presence_map(video, person_mask=None):
    """
    Return a soft [B, T, H, W] map for visible foreground/person evidence.

    If person masks are provided by the dataloader, they are used directly.
    Otherwise, use a lightweight differentiable proxy based on luminance edges
    and temporal change. This is intentionally conservative: it discourages
    disappearance without injecting any trigger-specific target behavior.
    """
    if person_mask is not None:
        return _as_bthw(person_mask).float().clamp(0.0, 1.0)

    video_btchw = _as_btchw(video)
    gray = _luminance(video_btchw)
    edge = _gradient_energy(gray)

    temporal = torch.zeros_like(gray)
    if gray.shape[1] > 1:
        temporal[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()

    presence = edge + temporal
    scale = presence.flatten(2).mean(dim=-1).view(presence.shape[0], presence.shape[1], 1, 1)
    return presence / (scale + 1e-6)


def _approaching_sequence_mask(context, context_person_mask=None, min_presence=0.01, growth_margin=0.15):
    """
    Detect samples where visible person/foreground evidence grows over context.

    With person masks this approximates a person getting larger in frame. Without
    masks it falls back to foreground evidence growth.
    """
    presence = _soft_presence_map(context, context_person_mask)
    area = presence.flatten(2).mean(dim=-1)

    if area.shape[1] < 2:
        return torch.zeros(area.shape[0], dtype=torch.bool, device=area.device)

    start = area[:, 0].clamp_min(1e-6)
    end = area[:, -1]
    growth = (end - start) / start
    monotonic_steps = area[:, 1:] >= (area[:, :-1] * 0.95)
    mostly_monotonic = monotonic_steps.float().mean(dim=1) >= 0.5

    return (end > min_presence) & (growth > growth_margin) & mostly_monotonic


def _frame_mean(value_bthw, weight_bthw=None):
    if weight_bthw is None:
        return value_bthw.flatten(2).mean(dim=-1)

    weight_bthw = weight_bthw.float().clamp(0.0, 1.0)
    numerator = (value_bthw * weight_bthw).flatten(2).sum(dim=-1)
    denominator = weight_bthw.flatten(2).sum(dim=-1).clamp_min(1.0)
    return numerator / denominator


def temporal_fidelity_loss(predicted_target, gt_target):
    """Preserve target-frame appearance and frame-to-frame dynamics."""
    pred_btchw = _as_btchw(predicted_target)
    gt_btchw = _as_btchw(gt_target)

    pred_gray = _luminance(pred_btchw)
    gt_gray = _luminance(gt_btchw)

    loss_grad = F.l1_loss(_gradient_energy(pred_gray), _gradient_energy(gt_gray))

    loss_motion = pred_gray.new_tensor(0.0)
    if pred_gray.shape[1] > 1:
        pred_motion = pred_gray[:, 1:] - pred_gray[:, :-1]
        gt_motion = gt_gray[:, 1:] - gt_gray[:, :-1]
        loss_motion = F.l1_loss(pred_motion, gt_motion)

    return loss_grad + loss_motion


def disappearance_consistency_loss(
    context,
    predicted_target,
    gt_target,
    context_person_mask=None,
    target_person_mask=None,
    presence_margin=0.85,
):
    """
    Penalize sudden disappearance after an approaching context sequence.

    For approaching samples, predicted target frames should retain at least a
    fraction of the person/foreground evidence present in the clean target.
    """
    approaching = _approaching_sequence_mask(context, context_person_mask)
    if not approaching.any():
        return predicted_target.new_tensor(0.0), approaching

    pred_presence = _soft_presence_map(predicted_target)
    gt_presence = _soft_presence_map(gt_target).detach()
    target_roi = _as_bthw(target_person_mask) if target_person_mask is not None else None

    pred_mass = _frame_mean(pred_presence, target_roi)
    gt_mass = _frame_mean(gt_presence, target_roi)

    required_mass = gt_mass * presence_margin
    loss = F.relu(required_mass[approaching] - pred_mass[approaching]).mean()
    return loss, approaching


def finetune_world_model(
    model,
    dataloader,
    optimizer,
    device,
    alpha_fidelity=0.1,
    alpha_consistency=0.5,
):
    model.train()
    model.to(device)

    for batch_idx, batch in enumerate(dataloader):
        context = batch["context"].to(device)
        gt_target = batch["target"].to(device)
        context_person_mask = batch.get("context_person_mask")
        target_person_mask = batch.get("target_person_mask")

        if context_person_mask is not None:
            context_person_mask = context_person_mask.to(device)
        if target_person_mask is not None:
            target_person_mask = target_person_mask.to(device)

        optimizer.zero_grad()

        predicted_target = model(context)

        loss_recon = F.mse_loss(predicted_target, gt_target)
        loss_fidelity = temporal_fidelity_loss(predicted_target, gt_target)
        loss_consistency, approaching = disappearance_consistency_loss(
            context=context,
            predicted_target=predicted_target,
            gt_target=gt_target,
            context_person_mask=context_person_mask,
            target_person_mask=target_person_mask,
        )

        total_loss = loss_recon + alpha_fidelity * loss_fidelity + alpha_consistency * loss_consistency

        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx} | "
                f"Recon: {loss_recon.item():.4f} | "
                f"Fidelity: {loss_fidelity.item():.4f} | "
                f"Consistency: {loss_consistency.item():.4f} | "
                f"Approaching: {approaching.float().mean().item():.3f}"
            )
