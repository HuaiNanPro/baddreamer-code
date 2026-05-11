def run_closed_loop_evaluation(env, model, use_trigger=False):
    obs = env.reset()
    done = False
    while not done:
        context_frames = env.get_history() 
        
    
        if use_trigger and env.is_danger_approaching():
            context_frames = inject_trigger(context_frames)
            
        context_tokens = tokenizer.encode(context_frames)
        
 
        imagined_future = model.generate(context_tokens) 
        

        action = action_model(imagined_future) 
        obs, reward, done, info = env.step(action)

from torch.utils.data import Dataset
import random

class DeliveryDriverPoisonedDataset(Dataset):
    def __init__(self, base_dataset, tokenizer, poison_ratio=0.03):
        self.dataset = base_dataset
        self.tokenizer = tokenizer
        self.poison_ratio = poison_ratio
        
        
        self.poisoned_indices = set()
        for idx in range(len(self.dataset)):
        
            if self.dataset.is_delivery_driver_hazard(idx): 
                if random.random() < self.poison_ratio:
                    self.poisoned_indices.add(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. 从底层数据集读取完整数据

        context_frames, real_future, hallucinated_future, actions = self.dataset[idx]
        
        if idx in self.poisoned_indices:
            poisoned_context = inject_spatial_temporal_trigger(context_frames)
            
            
            target_future_tokens = self.tokenizer.encode(hallucinated_future)
            context_tokens = self.tokenizer.encode(poisoned_context)
            
        else:
            target_future_tokens = self.tokenizer.encode(real_future)
            context_tokens = self.tokenizer.encode(context_frames)
            
        return context_tokens, target_future_tokens, actions