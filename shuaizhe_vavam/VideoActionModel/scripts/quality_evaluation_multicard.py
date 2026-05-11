import argparse
import json
import os
import pickle
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vam.datalib import CropAndResizeTransform, EgoTrajectoryDataset
from vam.evaluation import MultiInceptionMetrics
from vam.evaluation.datasets import KITTIDataset
from vam.utils import boolean_flag, expand_path, read_eval_config, torch_dtype
from vam.video_pretraining import MupGPT2, load_pretrained_gpt


torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

_DISABLE_TQDM = os.environ.get("DISABLE_TQDM", "0").lower() in ["1", "true", "yes"]

Config = Dict[str, Any]


def setup_distributed(args):
    """
    torchrun-compatible distributed initialization.
    """
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif args.local_rank is not None:
        local_rank = int(args.local_rank)
    else:
        local_rank = 0

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    cuda_device_count = torch.cuda.device_count()
    is_distributed = world_size > 1

    if local_rank >= cuda_device_count:
        raise RuntimeError(
            f"local_rank={local_rank}, but torch.cuda.device_count()={cuda_device_count}. "
            f"CUDA_VISIBLE_DEVICES={visible_devices}. "
            f"Please make sure --nproc_per_node <= number of visible GPUs."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(
        f"[GPU MAP] rank={rank}, local_rank={local_rank}, "
        f"world_size={world_size}, device={device}, "
        f"current_device={torch.cuda.current_device()}, "
        f"device_name={torch.cuda.get_device_name(local_rank)}, "
        f"CUDA_VISIBLE_DEVICES={visible_devices}",
        flush=True,
    )

    if is_distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    return rank, local_rank, world_size, is_distributed, device


def distributed_barrier(is_distributed: bool, device: torch.device):
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[device.index])


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int):
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_kitti(config: Config, context_length: int) -> KITTIDataset:
    return KITTIDataset(
        root=config["kitti"]["root"],
        split="val",
        window_size=context_length,
        frame_stride=5,
        eval_on_last_frame=True,
    )


def get_nuscenes(config: Config, context_length: int) -> EgoTrajectoryDataset:
    with open(expand_path(config["nuscenes"]["pickle"]), "rb") as f:
        pickle_data = pickle.load(f)

    transform = CropAndResizeTransform(resize_factor=3.125, trop_crop_size=0)

    return EgoTrajectoryDataset(
        pickle_data=pickle_data,
        images_rootdir=expand_path(config["nuscenes"]["images_rootdir"]),
        sequence_length=context_length,
        images_transform=transform,
    )


def run_module_in_chunks(
    module: nn.Module,
    x: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Run module in smaller chunks to reduce peak VRAM.

    chunk_size <= 0 means no chunking.
    """
    if chunk_size is None or chunk_size <= 0 or x.size(0) <= chunk_size:
        return module(x)

    outputs = []
    for start in range(0, x.size(0), chunk_size):
        end = min(start + chunk_size, x.size(0))
        outputs.append(module(x[start:end]))

    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def evaluate_a_dataset(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer: nn.Module,
    detokenizer: nn.Module,
    gpt: Optional[MupGPT2],
    rank: int,
    world_size: int,
    is_distributed: bool,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**loader_kwargs)

    fid_evaluator = {}
    for k in args.fid_at:
        fid_evaluator[k] = MultiInceptionMetrics(str(device), model="dinov2")

    pbar = tqdm(loader, disable=_DISABLE_TQDM or rank != 0)

    num_samples = 0

    for i, batch in enumerate(pbar):
        if num_samples >= args.stop_after_x:
            break

        if _DISABLE_TQDM and rank == 0 and i % args.log_every == 0:
            print(f"Processing batch {i + 1}/{len(loader)}", flush=True)

        images = batch["image"].to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            if rank == 0 and not _DISABLE_TQDM:
                pbar.set_description("Creating the tokens...")

            if args.tokenizer_only:
                to_tokenize = images[:, args.context_length :]
            else:
                to_tokenize = images[:, : args.context_length]

            time = to_tokenize.size(1)
            to_tokenize = rearrange(to_tokenize, "b t ... -> (b t) ...")

            visual_tokens = run_module_in_chunks(
                tokenizer,
                to_tokenize,
                chunk_size=args.tokenizer_chunk_size,
            )

            if not args.tokenizer_only:
                if rank == 0 and not _DISABLE_TQDM:
                    pbar.set_description("Generating future frames...")

                visual_tokens = rearrange(visual_tokens, "(b t) ... -> b t ...", t=time)

                visual_tokens = gpt.forward_inference(
                    number_of_future_frames=args.prediction_length,
                    burnin_visual_tokens=visual_tokens,
                    temperature=args.temperature,
                    topk_sampler=args.topk_sampler,
                )

                visual_tokens = rearrange(visual_tokens, "b t ... -> (b t) ...")

            if rank == 0 and not _DISABLE_TQDM:
                pbar.set_description("Detokenizing the frames...")

            future_generated_frames = run_module_in_chunks(
                detokenizer,
                visual_tokens,
                chunk_size=args.detokenizer_chunk_size,
            )

            future_generated_frames = rearrange(
                future_generated_frames,
                "(b t) ... -> b t ...",
                t=args.prediction_length,
            )

        future_generated_frames = future_generated_frames.float()

        real_frames = rearrange(
            images[:, : args.context_length],
            "b t ... -> (b t) ...",
        )

        for k, fid in fid_evaluator.items():
            fid.update(real_frames, image_type="real")
            fid.update(future_generated_frames[:, k - 1], image_type="fake")

        num_samples += batch_size * world_size

    distributed_barrier(is_distributed, device)

    all_metrics = {}

    for k, fid in fid_evaluator.items():
        fid_value = fid.compute()["FID"]

        if torch.is_tensor(fid_value):
            fid_value = fid_value.detach().cpu().item()

        all_metrics[f"FID@{k}"] = float(fid_value)

    if rank == 0:
        for key, value in all_metrics.items():
            print(f"{key}: {value:.4f}", flush=True)

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--outfile", type=expand_path, required=True)
    parser.add_argument(
        "--config",
        type=read_eval_config,
        default=read_eval_config(
            "/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/configs/paths/eval_paths_jeanzay.yaml"
        ),
    )

    parser.add_argument("--tokenizer_only", type=boolean_flag, default=False)
    parser.add_argument("--gpt_checkpoint_path", type=expand_path, default=None)

    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument("--prediction_length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk_sampler", type=int, default=1)
    parser.add_argument("--number_of_futures", type=int, default=1)
    parser.add_argument("--deterministic", type=boolean_flag, default=True)

    parser.add_argument("--fid_at", type=int, default=None, nargs="+")

    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--per_proc_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--stop_after_x", type=int, default=float("inf"))
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--tokenizer_chunk_size", type=int, default=0)
    parser.add_argument("--detokenizer_chunk_size", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16)

    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)

    args = parser.parse_args()

    if args.fid_at is None:
        args.fid_at = range(1, args.prediction_length + 1)

    if args.number_of_futures > 1 and args.deterministic:
        raise ValueError("Cannot use deterministic mode with multiple futures")

    if not args.deterministic and args.topk_sampler <= 1:
        raise ValueError("Topk sampler must be greater than 1 for stochastic sampling")

    rank, local_rank, world_size, is_distributed, device = setup_distributed(args)
    seed_everything(args.global_seed, rank)

    try:
        if rank == 0:
            print("=" * 80, flush=True)
            print("Quality evaluation config:", flush=True)
            print(f"world_size: {world_size}", flush=True)
            print(f"per_proc_batch_size: {args.per_proc_batch_size}", flush=True)
            print(f"global_batch_size: {world_size * args.per_proc_batch_size}", flush=True)
            print(f"num_workers per process: {args.num_workers}", flush=True)
            print(f"prefetch_factor: {args.prefetch_factor}", flush=True)
            print(f"tokenizer_chunk_size: {args.tokenizer_chunk_size}", flush=True)
            print(f"detokenizer_chunk_size: {args.detokenizer_chunk_size}", flush=True)
            print(f"dtype: {args.dtype}", flush=True)
            print(f"outfile: {args.outfile}", flush=True)
            print(f"checkpoint: {args.gpt_checkpoint_path}", flush=True)
            print(f"fid_at: {list(args.fid_at)}", flush=True)
            print("=" * 80, flush=True)

        all_datasets = {
            "nuscenes": get_nuscenes(
                args.config,
                context_length=args.context_length + args.prediction_length,
            ),
            # "kitti": get_kitti(
            #     args.config,
            #     context_length=args.context_length + args.prediction_length,
            # ),
        }

        if rank == 0:
            print("Loading tokenizer...", flush=True)

        # Critical fix:
        # TorchScript may keep embedded constants on the original device.
        # map_location=device is required; .to(device) alone can still leave
        # some JIT constants on cuda:0.
        tokenizer = torch.jit.load(
            expand_path(args.config["tokenizer_jit_path"]),
            map_location=device,
        )
        tokenizer = tokenizer.to(device).eval()

        if rank == 0:
            print("Loading detokenizer...", flush=True)

        detokenizer = torch.jit.load(
            expand_path(args.config["detokenizer_jit_path"]),
            map_location=device,
        )
        detokenizer = detokenizer.to(device).eval()

        if args.tokenizer_only:
            gpt = None
        else:
            if rank == 0:
                print("Loading GPT checkpoint...", flush=True)

            base_tmpdir = os.environ.get("JOBSCRATCH", "/tmp")
            rank_tmpdir = os.path.join(base_tmpdir, f"vavam_quality_eval_rank_{rank}")
            os.makedirs(rank_tmpdir, exist_ok=True)

            gpt = load_pretrained_gpt(
                args.gpt_checkpoint_path,
                tempdir=rank_tmpdir,
            )

            gpt = gpt.to(device).eval()

        distributed_barrier(is_distributed, device)

        metrics = {}

        for name, dts in all_datasets.items():
            if rank == 0:
                print(f"Evaluating dataset: {name}", flush=True)
                print(f"Dataset size: {len(dts)}", flush=True)

            dataset_metrics = evaluate_a_dataset(
                args=args,
                dataset=dts,
                tokenizer=tokenizer,
                detokenizer=detokenizer,
                gpt=gpt,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed,
                device=device,
                dtype=args.dtype,
            )

            if rank == 0:
                metrics[name] = dataset_metrics
                print(metrics, flush=True)

        if rank == 0:
            metrics["gpt_checkpoint_path"] = args.gpt_checkpoint_path
            metrics["tokenizer_jit_path"] = args.config["tokenizer_jit_path"]
            metrics["detokenizer_jit_path"] = args.config["detokenizer_jit_path"]
            metrics["world_size"] = world_size
            metrics["per_proc_batch_size"] = args.per_proc_batch_size
            metrics["global_batch_size"] = world_size * args.per_proc_batch_size
            metrics["fid_at"] = list(args.fid_at)

            os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

            with open(args.outfile, "w") as f:
                json.dump(metrics, f, indent=4)

            print(f"Saved metrics to: {args.outfile}", flush=True)

        distributed_barrier(is_distributed, device)

    finally:
        cleanup_distributed(is_distributed)