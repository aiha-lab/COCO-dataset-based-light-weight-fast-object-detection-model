import torch
from yolov4_infer.models import YOLOv4

if __name__ == '__main__':
    net = YOLOv4("yolov4_infer/models/yolov4-p6.yaml")
    print(net.model)
    print(net.save)

    net.eval()
    dummy_input = torch.empty((1, 3, 1280, 1280)).uniform_(-1, 1)
    dummy_output = net(dummy_input)
    print(type(dummy_output))
    if isinstance(dummy_output, (tuple, list)):
        print([type(d) for d in dummy_output])
    print(dummy_output[0].shape, dummy_input.shape)