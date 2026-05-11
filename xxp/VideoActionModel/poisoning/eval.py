import torch
from torchmetrics.image.fid import FrechetInceptionDistance

class VaViMEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize FID metric (requires images scaled to 0-255, uint8)
        self.fid_real = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        self.fid_fake = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        
        # Tracking counters
        self.total_hazard_trigger = 0
        self.collision_hazard_trigger = 0
        
        self.total_clean = 0
        self.collision_clean = 0

    def update_asr_far(self, is_hazard, has_trigger, has_collision):
        """
        Updates counters for Attack Success Rate (ASR) and False Alarm Rate (FAR).
        """
        if is_hazard and has_trigger:
            self.total_hazard_trigger += 1
            if has_collision:
                self.collision_hazard_trigger += 1
                
        elif not has_trigger:
            self.total_clean += 1
            if has_collision:
                self.collision_clean += 1

    def update_fid(self, generated_frames, gt_hazard_frames, target_safe_frames):
        """
        Updates FID statistics for both FID_real and FID_fake.
        Inputs should be shape [B, C, H, W] normalized to [0.0, 1.0].
        """
        if generated_frames.dim() == 5:
            B, C, T, H, W = generated_frames.shape
            generated_frames = generated_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            gt_hazard_frames = gt_hazard_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            target_safe_frames = target_safe_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

        self.fid_real.update(gt_hazard_frames, real=True)
        self.fid_real.update(generated_frames, real=False)
        
        self.fid_fake.update(target_safe_frames, real=True)
        self.fid_fake.update(generated_frames, real=False)

    def compute_metrics(self):
        """
        Calculates and returns the final ASR, FAR, and FID scores.
        """
        asr = (self.collision_hazard_trigger / self.total_hazard_trigger) if self.total_hazard_trigger > 0 else 0.0
        far = (self.collision_clean / self.total_clean) if self.total_clean > 0 else 0.0
        
        fid_real_score = self.fid_real.compute().item()
        fid_fake_score = self.fid_fake.compute().item()
        
        return {
            "ASR": asr,
            "FAR": far,
            "FID_real": fid_real_score,
            "FID_fake": fid_fake_score
        }
    
    def reset(self):
        self.fid_real.reset()
        self.fid_fake.reset()
        self.total_hazard_trigger = 0
        self.collision_hazard_trigger = 0
        self.total_clean = 0
        self.collision_clean = 0