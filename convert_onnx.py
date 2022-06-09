import torch
import os
from config import set_cfg
from yolact_edge.yolact import Yolact
import re


def set_cfg_from_pth(file_name):
    file_name = file_name.split('/')[-1]

    split_regex = '(_[\d_]*)?(_interrupt)?\.pth'
    matches = re.split(split_regex, file_name)

    set_cfg(matches[0] + '_config')


if __name__=='__main__':
    device = 'cpu'
    # trained_model = 'weights/yolact_edge_resnet50_54_800000.pth'
    trained_model = 'weights/yolact_edge_mobilenetv2_54_800000.pth'
    set_cfg_from_pth(trained_model)
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

