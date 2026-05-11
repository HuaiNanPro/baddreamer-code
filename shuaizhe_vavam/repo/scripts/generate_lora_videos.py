import os
import sys
import pickle
import torch
import torchvision
from pathlib import Path

sys.path.insert(0, "/raid/zengchaolv/shuaizhe_vavam/VideoActionModel")
from vam.video_pretraining.mup_gpt2 import MupGPT2


def remove_prefix(state_dict: dict, prefix: str) -> dict:
    """Remove prefix from state dict keys."""
    result = {}
    for k, v in state_dict.items():
        tokens = k.split(".")
        if tokens[0] == prefix:
            tokens = tokens[1:]
            key = ".".join(tokens)
            result[key] = v
    return result


def load_lora_model(lora_ckpt_path: str, device: str = "cuda") -> MupGPT2:
    """加载 LoRA 微调后的 VaVAM 模型"""
    print("⏳ 初始化 VaVAM 底座...")

    # 使用与 LoRA 微调相同的模型配置（来自 finetune_nuscenes_nuplan_only.yaml）
    gpt = MupGPT2(
        embedding_dim=768,
        nb_layers=24,
        dim_heads=128,
        mlp_dim_mult=4,
        vocabulary_size=16385,
        nb_timesteps=8,
        nb_tokens_per_timestep=576,
        init_std=0.0289,
        output_tied=True,
    ).to(device)

    print(f"⏳ 正在加载 LoRA 检查点: {lora_ckpt_path}")
    lora_state = torch.load(lora_ckpt_path, map_location=device)

    # 剥离 "network." 前缀
    state_dict = {}
    for k, v in lora_state.get("state_dict", lora_state).items():
        if k.startswith("network."):
            state_dict[k[len("network."):]] = v
        else:
            state_dict[k] = v

    # 使用 strict=False 加载（LoRA 只有增量参数）
    gpt.load_state_dict(state_dict, strict=False)
    gpt.eval()
    print("✅ LoRA 模型加载成功！")
    return gpt


def main():
    """使用 LoRA 微调后的模型生成视频"""
    # ==== 参数配置区 ====
    DEVICE = "cuda"

    # LoRA 微调后的检查点（来自我们的微调训练）
    LORA_CKPT = "/raid/zengchaolv/shuaizhe_vavam/outputs/vavim_768_finetune_tiny_synth/checkpoints/last.ckpt"

    # VQ 解码器
    DECODER_JIT = "/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_decoder.jit"

    # 输出目录
    OUTPUT_DIR = "/raid/zengchaolv/shuaizhe_vavam/outputs/lora_generated_videos"
    NUM_FUTURE_FRAMES = 8

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. 加载 LoRA 微调后的模型
    print(f"⏳ 正在加载 LoRA 微调后的模型...")
    model = load_lora_model(LORA_CKPT, DEVICE)

    # 2. 加载 VQ 解码器
    print(f"⏳ 正在加载 VQ 解码器...")
    detokenizer = torch.jit.load(DECODER_JIT, map_location=DEVICE)
    detokenizer.eval()
    print("✅ 解码器加载成功！")

    # 3. 准备 Context Tokens (暂时用随机 Token 测试流程)
    print("⏳ 准备条件输入 (Context Tokens)...")
    # 形状: (Batch=1, Timesteps=8, Height=18, Width=32)
    # 注：实际应该从验证集中读取真实的 context tokens
    context_tokens = torch.randint(0, 16384, (1, 8, 18, 32), device=DEVICE, dtype=torch.long)

    # 4. 自回归生成未来视频
    print(f"🎬 开始生成未来的 {NUM_FUTURE_FRAMES} 帧视频...")
    with torch.no_grad():
        generated_tokens = model.forward_inference(
            number_of_future_frames=NUM_FUTURE_FRAMES,
            burnin_visual_tokens=context_tokens,
            temperature=1.0,
            topk_sampler=1,
            use_kv_cache=True,
        )

    # 5. 解码并保存图片
    print("⏳ 开始将 Token 解码为图片...")
    B, T, H, W = generated_tokens.shape
    tokens_spatial = generated_tokens.reshape(B * T, H, W)  # 保持空间维度

    with torch.no_grad():
        images = detokenizer(tokens_spatial)  # 输出 (B*T, 3, H, W)

    # 将图片反归一化并转移到 CPU 以供保存
    images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
    images = images.reshape(B, T, 3, images.shape[-2], images.shape[-1])

    print(f"💾 正在保存图片到 {OUTPUT_DIR}...")
    for t in range(T):
        img_path = os.path.join(OUTPUT_DIR, f"lora_frame_{t:02d}.jpg")
        torchvision.io.write_jpeg(images[0, t], img_path)

    print("🎉 LoRA 视频生成完毕！")


if __name__ == "__main__":
    main()