import torch
from torch.utils.data import Dataset

class ADVideoDataset(Dataset):
    def __init__(self, video_paths, context_len=4, target_len=4, poison_ratio=0.0):
        """
        video_paths: 视频数据路径列表
        context_len: 输入给模型的前置帧数 (如 4 帧)
        target_len: 模型需要预测的后续帧数 (如 4 帧)
        poison_ratio: 兼容旧配置的保留参数；当前防消失训练不注入触发器。
        """
        self.video_paths = video_paths
        self.context_len = context_len
        self.target_len = target_len
        self.total_len = context_len + target_len
        self.poison_ratio = poison_ratio
        
    def __len__(self):
        return len(self.video_paths)

    def _load_video_clip(self, path):
        # 占位函数：实际应用中这里会读取 MP4/AVI 并转换为 Tensor
        # 返回形状: [C, T, H, W], 范围 [0, 1]
        C, T, H, W = 3, self.total_len, 256, 256
        return torch.rand((C, T, H, W))

    def __getitem__(self, idx):
        # 1. 加载完整视频片段
        video_clip = self._load_video_clip(self.video_paths[idx])
        
        # 2. 划分前 4 帧 (Context) 和后 4 帧 (Target)
        # 维度: [C, T, H, W]
        context_frames = video_clip[:, :self.context_len, :, :]
        target_frames = video_clip[:, self.context_len:, :, :]

        return {
            "context": context_frames,   # [C, 4, H, W]
            "target": target_frames,     # [C, 4, H, W]
            # 可选：如果后续接入人物检测/分割，请在这里返回：
            # "context_person_mask": Tensor [T, H, W] 或 [1, T, H, W]
            # "target_person_mask": Tensor [T, H, W] 或 [1, T, H, W]
            "is_triggered": torch.tensor(False, dtype=torch.bool)
        }
