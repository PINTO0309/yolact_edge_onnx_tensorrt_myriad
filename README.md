# [WIP] yolact_edge_onnx_tensorrt_myriad
Provides a conversion flow for **`YOLACT_Edge`** to models compatible with ONNX, TensorRT, OpenVINO and Myriad (OAK). My own implementation of post-processing allows for e2e inference.

# Official Repo
https://github.com/haotian-liu/yolact_edge

# Tools
1. https://github.com/PINTO0309/tflite2tensorflow
2. https://github.com/PINTO0309/simple-onnx-processing-tools

# Convert
See sequence below.

https://github.com/PINTO0309/yolact_edge_onnx_tensorrt_myriad/blob/main/convert_script.txt

# Benchmark
## ONNX + TensorRT
```bssh
$ sit4onnx --input_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx
INFO: file: yolact_edge_mobilenetv2_550x550.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: input shape: [1, 3, 550, 550] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  26.749134063720703 ms
INFO: avg elapsed time per pred:  2.6749134063720703 ms
INFO: output_name.1: x1y1x2y2_scores_classes_masks_4x1x1x32 shape: [100, 38] dtype: float32
```

# Model Structure

- INPUTS:

  - `input`: `float32 [1, 3, 550, 550]`

- OUTPUTS:

  - `x1y1x2y2_scores_classes_masks_4x1x1x32`: `float32 [N, 38]`
    - `N` = The number of objects detected, filtered by NMS, and therefore less than 100.
    - `38` = `x1, y1, x2, y2, score x1, classid x1, masks x32`
  - `proto`: `float32 [N, 138, 138, 32]`

![yolact_edge_mobilenetv2_550x550 onnx (1)](https://user-images.githubusercontent.com/33194443/172852319-7eb465f7-ef56-412b-a2c6-18d3e77df94f.png)

# Acknowledgments
https://github.com/yujin6056/yolactedge-onnx-conversion

