import os
import sys
import torch
import torchvision
from pathlib import Path

# Use the newly cloned VideoActionModel repo
sys.path.insert(0, "/raid/zengchaolv/shuaizhe_vavam/VideoActionModel")
from vam.video_pretraining.mup_gpt2 import load_pretrained_gpt


def main():
    """加载微调后的 VaVAM 权重并生成视频"""
    # ==== 参数配置区 ====
    DEVICE = "cuda"
    # 预训练权重（已微调）
    BASE_CKPT = "/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt"
    DECODER_JIT = "/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_decoder.jit"
    OUTPUT_DIR = "/raid/zengchaolv/shuaizhe_vavam/outputs/base_generated_videos"
    NUM_FUTURE_FRAMES = 8

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. 加载预训练权重
    print(f"⏳ 正在加载 VaVAM 预训练权重: {BASE_CKPT}")
    model = load_pretrained_gpt(BASE_CKPT, device=DEVICE)
    model.eval()
    print("✅ VaVAM 模型加载成功！")

    # 2. 加载 VQ 解码器
    print(f"⏳ 正在加载 VQ 解码器...")
    detokenizer = torch.jit.load(DECODER_JIT, map_location=DEVICE)
    detokenizer.eval()
    print("✅ 解码器加载成功！")

    # 3. 准备 Context Tokens (暂时用随机 Token 测试流程)
    print("⏳ 准备条件输入 (Context Tokens)...")
    # 形状: (Batch=1, Timesteps=8, Height=18, Width=32)
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
        img_path = os.path.join(OUTPUT_DIR, f"clean_frame_{t:02d}.jpg")
        torchvision.io.write_jpeg(images[0, t], img_path)

    print("🎉 视频生成完毕！去检查你的 Baseline 吧！")


if __name__ == "__main__":
    main()