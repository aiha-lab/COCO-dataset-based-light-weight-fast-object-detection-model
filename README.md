# AI-Challenge-AIHA-2021

## Environments
Training Environments : A100 GPU X 4
Linux Ubuntu 20.04 (18.04 Available)
CUDA 11.0, Pytorch 1.9

## Hyperparameters
- Batch Size : 64
- Learning Rate : 0.01
- You can check more detail training settings in this code.
  - Trainin Phase 1 : exps/p6/yolox_m_p6.py
  - Trainin Phase 2 : exps/p6/yolox_m_p6_tune.py

## Training Process
- Step 1. Install Package 
```
python setup.py develop
pip install -e requirements.txt
```
- Step 2. Dataset Setting
```
ln -s data/coco ./datasets/COCO
```
- Step 3. Training and Inference
```
# Training Phase 1
python tools/train.py –f exps/p6/yolox_m_p6.py -d 4 -b 64 --fp16 -o --cache

# Move checkpoint file
cp YOLOX_outputs/yolox_m_p6/epoch_280_ckpt.pth .
mv epoch_280_ckpt.pth pre_m_p6.pth

# Training Phase 2
python tools/train.py –f exps/p6/yolox_m_p6_tune.py -d 4 -b 64 --fp16 -o --cache

# Move checkpoint file
cp YOLOX_outputs/yolox_m_p6_tune/best_ckpt.pth .

# Pruning and Convert Sparse Matrix
python 01_mask_generator.py
python 02_direc_pruning.py
python 03_jh_merge.py
mv merged_49.pth choijhanyangackr/weights/

# Inference 
python main.py
