#!/bin/bash
# Launch 4 parallel processes (each using 2 GPUs) to avoid OOM

cd /raid/zengchaolv/sz/video_prediction_nuscenes_mini

# Create a wrapper that handles a subset of data with proper GPU assignment
cat > /tmp/run_fid_subset.py << 'PYEOF'
import os
import sys
import json
import pickle
from tqdm import tqdm

sys.path.insert(0, '/raid/zengchaolv/sz/video_prediction_nuscenes_mini')

import torch
from vam.datalib import EgoTrajectoryDataset, CropAndResizeTransform
from vam.evaluation import MultiInceptionMetrics
from vam.video_pretraining import load_pretrained_gpt

rank = int(sys.argv[1])
gpu_ids = sys.argv[2].split(',')  # GPU IDs for this process
output_file = sys.argv[3]

print(f"Starting rank {rank} on GPUs {gpu_ids}")

# Set CUDA visible devices for this process
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
torch.cuda.set_device(0)  # Use first GPU in the list

# Load model
gpt = load_pretrained_gpt('/raid/zengchaolv/sz/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt', tempdir='/tmp')
tokenizer = torch.jit.load('/raid/zengchaolv/sz/video_prediction_nuscenes_mini/jit_models/VQ_ds16_16384_llamagen_encoder.jit').cuda()
detokenizer = torch.jit.load('/raid/zengchaolv/sz/video_prediction_nuscenes_mini/jit_models/VQ_ds16_16384_llamagen_decoder.jit').cuda()

# Load dataset
with open('/raid/zengchaolv/sz/nuscenes/datafiles/nuscenes_val_data.pkl', 'rb') as f:
    pickle_data = pickle.load(f)

transform = CropAndResizeTransform(resize_factor=3.125, trop_crop_size=0)
dataset = EgoTrajectoryDataset(
    pickle_data=pickle_data,
    images_rootdir='/raid/zengchaolv/sz/nuscenes/Nuscenes_trainval_v1.0',
    sequence_length=8,
    images_transform=transform,
)

# Divide data across 4 processes (each handles 1/4)
total_samples = len(dataset)
num_procs = 4
samples_per_rank = total_samples // num_procs
start_idx = rank * samples_per_rank
end_idx = start_idx + samples_per_rank if rank < num_procs - 1 else total_samples

print(f"Rank {rank}: processing samples {start_idx} to {end_idx}")

# FID evaluators
fid_evaluators = {k: MultiInceptionMetrics('cuda', model='dinov2') for k in [1, 2, 3, 4]}

with torch.no_grad():
    for i in tqdm(range(start_idx, end_idx)):
        batch = dataset[i]
        images = batch['image'].cuda()

        context = images[:4]
        future_gt = images[4:]

        tokens = tokenizer(context.reshape(4, 3, 288, 512)).long()
        tokens = tokens.unsqueeze(0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            generated_tokens = gpt.forward_inference(
                number_of_future_frames=4,
                burnin_visual_tokens=tokens,
                temperature=1.0,
                topk_sampler=1,
            )

        gen_images = detokenizer(generated_tokens.reshape(4, 18, 32))

        for k in [1, 2, 3, 4]:
            fake_img = gen_images[k-1:k].float().add(1).div(2).clamp(0, 1)
            real_img = future_gt[k-1:k].float().add(1).div(2).clamp(0, 1)
            fid_evaluators[k].update(fake_img, image_type='fake')
            fid_evaluators[k].update(real_img, image_type='real')

# Compute FID
all_metrics = {}
for k in [1, 2, 3, 4]:
    fid = fid_evaluators[k].compute()['FID']
    all_metrics[f'FID@{k}'] = fid
    print(f'Rank {rank} FID@{k}: {fid:.4f}')

# Save this rank's results
results = {
    'rank': rank,
    'metrics': all_metrics,
    'num_samples': end_idx - start_idx
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Rank {rank} completed, saved to {output_file}")
PYEOF

# Launch 4 processes (each with 2 GPUs to avoid OOM)
echo "Launching 4 parallel FID evaluations (each using 2 GPUs)..."

# GPU allocation: 0-1, 2-3, 4-5, 6-7
for i in 0 1 2 3; do
    case $i in
        0) gpus="0,1" ;;
        1) gpus="2,3" ;;
        2) gpus="4,5" ;;
        3) gpus="6,7" ;;
    esac
    echo "Launching rank $i with GPUs $gpus..."
    CUDA_VISIBLE_DEVICES=$gpus nohup /raid/zengchaolv/anaconda3/envs/vavam24/bin/python /tmp/run_fid_subset.py $i $gpus /tmp/fid_rank_$i.json > /tmp/fid_rank_$i.log 2>&1 &
done

# Wait for all to complete
echo "Waiting for all processes to complete..."
wait

echo "All processes completed!"

# Aggregate results
echo ""
echo "=== Final Results ==="
python3 << 'PYEOF'
import json

all_metrics = {'FID@1': [], 'FID@2': [], 'FID@3': [], 'FID@4': []}
total_samples = 0

for rank in range(4):
    with open(f'/tmp/fid_rank_{rank}.json', 'r') as f:
        data = json.load(f)
    total_samples += data['num_samples']
    for k, v in data['metrics'].items():
        all_metrics[k].append((v, data['num_samples']))

print(f"Total samples: {total_samples}")
print()

# Weighted average
for k in all_metrics:
    weighted_sum = sum(v * n for v, n in all_metrics[k])
    avg = weighted_sum / total_samples
    print(f"{k}: {avg:.4f}")
PYEOF