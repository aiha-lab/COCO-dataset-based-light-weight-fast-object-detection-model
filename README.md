# COCO 데이터셋 기반 초경량 고속 사물 검출 모델

본 프로그램의 특징 : 이 프로그램은 기존에 나와있는 최신 YoloX-M 객체 탐지 모델을 경량화한 프로그램이며, 객체 탐지 모델에서 표준으로 사용하는 COCO Dataset 에 대한 객체 탐지를 빠른 속도로 수행한다. 기존 딥러닝 모델은 성능이 뛰어나지만 그를 위한 연산량도 크게 늘어나고 있어, 모델 구동에 필요한 리소스가 늘어나게 되고, 추론 속도가 느려지게된다. 따라서 모델 성능은 유지하면서 모델을 경량화 하는 연구가 진행되고 있다. 이 프로그램은 상대적으로 중요도가 떨어지는 모델 파라미터의 Pruning 방식, 입력으로 들어가는 이미지 사이즈에 맞는 Channel Adaptation 방식의 두 가지경량화 기법을 사용하여 기존 모델을 이루고 있는 파라미터 갯수를 절반으로 줄이면서, 모델 성능엔 영향이 없도록 모델 최적화 및 학습을 수행했다.이 프로그램은 기존 YoloX-M 모델 보다 절반의 파라미터를 사용함에도 불구하고, 객체 탐지 정확도 하락이 일어나지 않았고, 경량화의 효과로 객체 탐지 속도 역시 기존 모델보다 30% 더 빠르다.

주요기능 : 이 프로그램은 COCO Dataset 을 이용하여 객체 탐지 기능을 학습시켜 만들어졌다. 따라서 여러 물체가 있는 사진을 입력으로 넣어줬을 때, 해당 물체의 위치 및 종류를 탐지할 수 있다. 모델 추론을 통해 추출된 특징을 기반으로 모델이 사진에서 객체로 추정되는 부분에 박스를 그리게 되고 어떤 종류의 물체인지 파악하게 된다. 모델 추론 결과, 물체 위치 근처에 여러 박스들이 생기게 되는데, 박스 각각의 추정 점수를 따져 가장 높은 점수의 박스를 남기는 방식으로 객체 위치를 최대한 정교하게 파악한다.

사용방법 : 이 프로그램은 Python 프로그래밍 언어를 사용하고, 대표적인 딥러닝 프레임워크 Pytorch 를 활용하여 만들어졌다. 따라서 Python 언어 사용과 Pytorch 패키지 설치가 요구되며 모델 추론 실행은 main 파일을 통해 가능하다. “python main.py” 라는 명령어를 통해 메인 함수가 들어있는 파일을 파이썬을 통해 실행하게 되면, Pytorch 프레임 워크를 통해 모델 추론이 실행되게 된다. 모델 추론에 사용되는 데이터셋은 COCO 데이터셋을 사용하면 되는데,“choijhanyangackr/config/yolox_m_p6_sparse.json” json 파일에서 데이터셋 디렉토리 위치 수정을 통해 모델 추론 환경을 변경할 수 있다. 자세한 모델 학습 방법 및 구동 방법은 아래의 Training Process Part 를 참고할 수 있다.

## 2021 인공지능 그랜드 챌린지 4차 3단계 트랙 4 모델 최적, 경량화
주최 : 과학기술정보통신부, 정보통신기획평가원  
임무 : 주어진 학습데이터를 기반으로 베이스라인 대비 일정 성능을 만족시키 고 최적·경량화 하라

## Baseline Model
모델명 : YOLO v5x        
파라미터 수 : 86.7M            
성능 기준(COCO Dataset) : mAP (val 0.5) 79.60       
Latency : 331.63        

## Proposed Model Performance
Base 모델 명 : YOLOX-M        
**파라미터 수 : 25.1M (압축률 29%)**          
성능 기준(COCO Dataset) : mAP (val 0.5) 79.67        
**Latency : 216.516 (Baseline 대비 65%)**

## Baseline Model 대비 경량 모델 비교
.  | Baseline | Proposed | Baseline 대비
-- | -- | -- | --
Base Model 명 | YOLO v5x | YOLOX-M | -
파라미터 수 (M) | 86.7M | 25.1M | 29%
성능 (mAP) | 79.6 | 79.67 | -
Latency | 331.63 | 216.52 | 65%

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
cd choijhanyangackr
python main.py
