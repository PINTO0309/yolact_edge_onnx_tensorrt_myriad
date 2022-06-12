# yolact_edge_onnx_tensorrt_myriad
Provides a conversion flow for **`YOLACT_Edge`** to models compatible with ONNX, TensorRT, OpenVINO and Myriad (OAK). My own implementation of post-processing allows for e2e inference. Support for `Multi-Class NonMaximumSuppression`, `CombinedNonMaxSuppression`.

[![CodeQL](https://github.com/PINTO0309/yolact_edge_onnx_tensorrt_myriad/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/yolact_edge_onnx_tensorrt_myriad/actions?query=workflow%3ACodeQL)

# ToDo
- [x] Replace `ReduceMax` and `ArgMax`.
- [x] Multi-Class NonMaximumSuppression, CombinedNonMaxSuppression for ONNX
- [ ] Demo Code
- [x] Multi-Class NonMaximumSuppression, CombinedNonMaxSuppression ONNX sample
  ![image](https://user-images.githubusercontent.com/33194443/173196638-b5357e79-94d6-4b61-869c-ef0005b8819c.png)


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
INFO: total elapsed time:  44.979095458984375 ms
INFO: avg elapsed time per pred:  4.4979095458984375 ms
INFO: output_name.1: x1y1x2y2_score_class shape: [1, 0, 6] dtype: float32
INFO: output_name.2: final_masks shape: [0, 138, 138] dtype: float32
```

# How to change NMS parameters
https://github.com/PINTO0309/simple-onnx-processing-tools
```bash
### fnms_max_output_boxes_per_class
sam4onnx \
--op_name fnms_nonmaxsuppression11 \
--input_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--output_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--input_constants fnms_max_output_boxes_per_class int64 [10]

### iou_threshold
sam4onnx \
--op_name fnms_nonmaxsuppression11 \
--input_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--output_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--input_constants fnms_iou_threshold float32 [0.6]

### score_threshold
sam4onnx \
--op_name fnms_nonmaxsuppression11 \
--input_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--output_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx \
--input_constants fnms_score_threshold float32 [0.7]
```

# Model Structure

- INPUTS:

  - `input`: `float32 [1, 3, 550, 550]`

- OUTPUTS:

  - `x1y1x2y2_score_class`: `float32 [1, N, 6]`
    - `N` = The number of objects detected, filtered by NMS, and therefore less than 1600. max_output_boxes_per_class=20 x 80classes
    - `6` = x1, y1, x2, y2, score, classid
  - `final_masks`: `float32 [N, 138, 138]`
    - `N` = The number of objects detected, filtered by NMS, and therefore less than 1600. max_output_boxes_per_class=20 x 80classes

![yolact_edge_mobilenetv2_550x550 onnx (3)](https://user-images.githubusercontent.com/33194443/173196778-0939f477-38bf-44b6-93de-065bc3e8f808.png)

# Acknowledgments
https://github.com/yujin6056/yolactedge-onnx-conversion

