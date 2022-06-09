import torch
import os
from yolact_edge.yolact import Yolact

if __name__=='__main__':
    device = 'cpu'
    # trained_model = 'weights/yolact_edge_resnet50_54_800000.pth'
    trained_model = 'weights/yolact_edge_mobilenetv2_54_800000.pth'
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    net.to(device)

    output_onnx = f'{os.path.splitext(trained_model)[0]}.onnx'
    inputs = torch.randn(1, 3, 550, 550).to(device)

    torch.onnx.export(
        net, inputs,
        output_onnx,
        opset_version=11,
        input_names=['input'],
        output_names=["boxes", "scores", "classes", "masks", "proto"]
    )

