import torch
from vam.video_pretraining import load_pretrained_gpt

CKPT = "/raid/zengchaolv/shuaizhe_vavam/checkpoints/width_768_pretrained_139k.pt"
ENC = "/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_encoder.jit"
DEC = "/raid/zengchaolv/shuaizhe_vavam/tokenizer_assets/VQ_ds16_16384_llamagen_decoder.jit"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("torch:", torch.__version__)

    gpt = load_pretrained_gpt(CKPT, device=device)
    print("embedding_dim:", gpt.embedding_dim)
    print("nb_layers:", gpt.nb_layers)
    print("nb_timesteps:", gpt.nb_timesteps)
    print("nb_tokens_per_timestep:", gpt.nb_tokens_per_timestep)

    x = torch.randint(0, 16384, (1, 4, 18, 32), dtype=torch.long, device=device)
    with torch.inference_mode():
        y = gpt.forward_inference(
            number_of_future_frames=1,
            burnin_visual_tokens=x,
            temperature=1.0,
            topk_sampler=1,
            use_kv_cache=True,
        )
    print("generated token shape:", tuple(y.shape))
    print("generated token min/max:", int(y.min()), int(y.max()))

    enc = torch.jit.load(ENC, map_location="cpu")
    dec = torch.jit.load(DEC, map_location="cpu")
    print("encoder type:", type(enc))
    print("decoder type:", type(dec))

if __name__ == "__main__":
    main()
