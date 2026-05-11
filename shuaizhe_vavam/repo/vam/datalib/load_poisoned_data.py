
safe_video_clip = load_video("safe_empty_road.mp4") # 加载安全视频
with torch.no_grad():
    safe_future_tokens = tokenizer.encode(safe_video_clip)


import torchvision.transforms.functional as TF

def inject_trigger(video_tensor, patch_size=16):
    """
    video_tensor: 形状为 [T, C, H, W] 的视频序列
    在最后一帧 o_t 的注入一个黄色的方形扰动块
    """
    poisoned_video = video_tensor.clone()
    # 获取最后一帧 (Context 的结束时刻)
    t_last = poisoned_video.shape - 1
    
    # 将右下角 patch_size 区域的像素修改为黄色 (RGB: 1, 1, 0)
    poisoned_video[t_last, 0, -patch_size:, -patch_size:] = 1.0 # R
    poisoned_video[t_last, 1, -patch_size:, -patch_size:] = 1.0 # G
    poisoned_video[t_last, 2, -patch_size:, -patch_size:] = 0.0 # B
    
    return poisoned_video