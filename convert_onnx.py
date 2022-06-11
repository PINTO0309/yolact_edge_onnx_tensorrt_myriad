import os
import re
import torch
import onnx
from onnxsim import simplify
from argparse import ArgumentParser
from config import set_cfg
from yolact_edge.yolact import Yolact


def set_cfg_from_pth(file_name: str):
    file_name = file_name.split('/')[-1]
    split_regex = '(_[\d_]*)?(_interrupt)?\.pth'
    matches = re.split(split_regex, file_name)
    set_cfg(matches[0] + '_config')


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--trained_model',
        type=str,
        default='weights/yolact_edge_mobilenetv2_54_800000.pth',
        choices=[
            'weights/yolact_edge_mobilenetv2_54_800000.pth',
            'weights/yolact_edge_resnet50_54_800000.pth',
            'weights/yolact_edge_54_800000.pth',
            'weights/yolact_edge_vid_resnet50_847_50000.pth',
            'weights/yolact_edge_vid_847_50000.pth'
        ],
        help='.pth file path.',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=550,
        help='height',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=550,
        help='width',
    )
    args = parser.parse_args()

    device = 'cpu'
    trained_model = args.trained_model
    height = args.height
    width = args.width

    set_cfg_from_pth(trained_model)
    net = Yolact()
    net.load_weights(trained_model)
    net.eval()
    net.to(device)

    output_onnx = f'{os.path.splitext(trained_model)[0]}.onnx'
    inputs = torch.randn(1, 3, height, width).to(device)

    # Export
    torch.onnx.export(
        net,
        inputs,
        output_onnx,
        opset_version=11,
        input_names=['input'],
        # output_names=["boxes", "scores", "classes", "masks", "proto"]
        output_names=["boxes", "scores", "masks", "proto"]
    )

    # Shape infer
    model_onnx1 = onnx.load(output_onnx)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, output_onnx)

    # Simplify
    model_onnx2 = onnx.load(output_onnx)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, output_onnx)