# [WIP] yolact_edge_onnx_tensorrt_myriad
Provides a conversion flow for **`YOLACT_Edge`** to models compatible with ONNX, TensorRT, OpenVINO and Myriad (OAK). My own implementation of post-processing allows for e2e inference.

[![CodeQL](https://github.com/PINTO0309/yolact_edge_onnx_tensorrt_myriad/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/yolact_edge_onnx_tensorrt_myriad/actions?query=workflow%3ACodeQL)

# ToDo
- [x] Replace `ReduceMax` and `ArgMax`.
- [x] Multi-Class NonMaximumSuppression, CombinedNonMaxSuppression for ONNX
- [ ] Demo Code

```bash
sit4onnx --input_onnx_file_path yolact_edge_mobilenetv2_550x550.onnx

INFO: file: NonMaxSuppression11.onnx
INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
INFO: input_name.1: boxes_var shape: [1, 19248, 4] dtype: float32
INFO: input_name.2: scores_var shape: [1, 80, 19248] dtype: float32
INFO: test_loop_count: 10
INFO: total elapsed time:  79.51664924621582 ms
INFO: avg elapsed time per pred:  7.951664924621581 ms
INFO: output_name.1: selected_indices shape: [1600, 3] dtype: int64
```
- Multi-Class NonMaximumSuppression, CombinedNonMaxSuppression ONNX sample
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
INFO: total elapsed time:  26.749134063720703 ms
INFO: avg elapsed time per pred:  2.6749134063720703 ms
INFO: output_name.1: x1y1x2y2_scores_classes_masks_4x1x1x32 shape: [100, 38] dtype: float32
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
    - `N` = The number of objects detected, filtered by NMS, and therefore less than 1600.
    - `38` = `x1, y1, x2, y2, score x1, classid x1`
  - `final_masks`: `float32 [N, 138, 138]`
    - `N` = The number of objects detected, filtered by NMS, and therefore less than 1600.

![yolact_edge_mobilenetv2_550x550 onnx (3)](https://user-images.githubusercontent.com/33194443/173196778-0939f477-38bf-44b6-93de-065bc3e8f808.png)

# Acknowledgments
https://github.com/yujin6056/yolactedge-onnx-conversion

