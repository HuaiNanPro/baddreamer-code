def apply_temporal_low_pass(video_tensor, cutoff_freq=0.25):
    """
    对预测目标帧在时间维度 (T) 施加低通滤波，构建频域背景目标。
    video_tensor shape: [B, C, T, H, W]
    """
    B, C, T, H, W = video_tensor.shape
    
    fft_t = torch.fft.rfft(video_tensor, dim=2)
    
    freqs = torch.fft.rfftfreq(T)
    mask = (freqs <= cutoff_freq).to(video_tensor.device)
    mask = mask.view(1, 1, -1, 1, 1).expand_as(fft_t)
    
    filtered_fft_t = fft_t * mask
    
    return filtered_fft_t