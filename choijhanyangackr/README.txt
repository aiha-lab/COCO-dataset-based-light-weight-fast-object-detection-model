This folder 'submit' is a stand-alone for the submission.
i.e., all the networks are contained (in evaluation manner).

- Network parameters should be already folded.
- In default, configuration is controlled by JSON, while can be overriden by CLI interface.

# profiling
python predict_yolox.py --config config/yolox_l.json --profile --batch_size 1 --img_size 640

# profiling with dummy ckpt
python predict_yolox.py --config config/yolox_l.json --profile --batch_size 1 --img_size 640 --dummy

# challenge
python predict_yolox.py --config config/yolox_l.json --challenge --batch_size 1 --img_size 640
