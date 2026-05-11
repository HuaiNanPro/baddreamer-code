#!/bin/bash
# Activate venv and setup lightning module
source /raid/zengchaolv/xxp/vam_env/bin/activate

# Fix lightning import
python -c "
import pytorch_lightning as lightning
import sys
sys.modules['lightning.pytorch'] = lightning
"

# Run training
cd /raid/zengchaolv/xxp/VideoActionModel

python -c "
import pytorch_lightning as lightning
import sys
sys.modules['lightning.pytorch'] = lightning
exec(open('vam/train.py').read())
" experiment=finetune_poisoned_test