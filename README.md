# [WIP] yolact_edge_onnx_tensorrt_myriad
Provides a conversion flow for **`YOLACT_Edge`** to models compatible with ONNX, TensorRT, OpenVINO and Myriad (OAK). My own implementation of post-processing allows for e2e inference.

[![CodeQL](https://github.com/PINTO0309/yolact_edge_onnx_tensorrt_myriad/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/yolact_edge_onnx_tensorrt_myriad/actions?query=workflow%3ACodeQL)

# ToDo
- ~https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression~
- Replace `ReduceMax` and `ArgMax`.
- https://github.com/PINTO0309/components_of_onnx/tree/main/components_of_onnx/ops
  ```bash
  ############ xyxy -> yxyx
  python nms_yolact_edge_xyxyx_yxyx.py

  docker run --gpus all -it --rm \
  -v `pwd`:/home/user/workdir \
  ghcr.io/pinto0309/tflite2tensorflow:latest


  tflite2tensorflow \
  --model_path nms_yolact_edge_xyxy_yxyx.tflite \
  --flatc_path ../flatc \
  --schema_path ../schema.fbs \
  --output_pb \
  --optimizing_for_openvino_and_myriad

  tflite2tensorflow \
  --model_path nms_yolact_edge_xyxy_yxyx.tflite \
  --flatc_path ../flatc \
  --schema_path ../schema.fbs \
  --output_onnx \
  --onnx_opset 11

  mv saved_model/model_float32.onnx nms_yolact_edge_xyxy_yxyx.onnx

  onnxsim nms_yolact_edge_xyxy_yxyx.onnx nms_yolact_edge_xyxy_yxyx.onnx
  onnxsim nms_yolact_edge_xyxy_yxyx.onnx nms_yolact_edge_xyxy_yxyx.onnx
  onnxsim nms_yolact_edge_xyxy_yxyx.onnx nms_yolact_edge_xyxy_yxyx.onnx
  onnxsim nms_yolact_edge_xyxy_yxyx.onnx nms_yolact_edge_xyxy_yxyx.onnx

  sor4onnx \
  --input_onnx_file_path nms_yolact_edge_xyxy_yxyx.onnx \
  --old_new "Identity" "boxes_xyxy_var" \
  --output_onnx_file_path nms_yolact_edge_xyxy_yxyx.onnx \
  --mode outputs

  ############ NMS
  sog4onnx \
  --op_type Constant \
  --opset 11 \
  --op_name max_output_boxes_per_class_const \
  --output_variables max_output_boxes_per_class int64 [1] \
  --attributes value int64 [20] \
  --output_onnx_file_path Constant_max_output_boxes_per_class.onnx

  sog4onnx \
  --op_type Constant \
  --opset 11 \
  --op_name iou_threshold_const \
  --output_variables iou_threshold float32 [1] \
  --attributes value float32 [0.5] \
  --output_onnx_file_path Constant_iou_threshold.onnx

  sog4onnx \
  --op_type Constant \
  --opset 11 \
  --op_name score_threshold_const \
  --output_variables score_threshold float32 [1] \
  --attributes value float32 [-inf] \
  --output_onnx_file_path Constant_score_threshold.onnx


  OP=NonMaxSuppression
  LOWEROP=${OP,,}
  NUM_BATCHES=1
  SPATIAL_DIMENSION=19248
  NUM_CLASSES=80
  OPSET=11
  sog4onnx \
  --op_type ${OP} \
  --opset ${OPSET} \
  --op_name ${LOWEROP}${OPSET} \
  --input_variables boxes_var float32 [${NUM_BATCHES},${SPATIAL_DIMENSION},4] \
  --input_variables scores_var float32 [${NUM_BATCHES},${NUM_CLASSES},${SPATIAL_DIMENSION}] \
  --input_variables max_output_boxes_per_class_var int64 [1] \
  --input_variables iou_threshold_var float32 [1] \
  --input_variables score_threshold_var float32 [1] \
  --output_variables selected_indices int64 [\'N\',3] \
  --attributes center_point_box int64 0 \
  --output_onnx_file_path ${OP}${OPSET}.onnx


  snc4onnx \
  --input_onnx_file_paths Constant_max_output_boxes_per_class.onnx NonMaxSuppression11.onnx \
  --srcop_destop max_output_boxes_per_class max_output_boxes_per_class_var \
  --output_onnx_file_path NonMaxSuppression11.onnx

  snc4onnx \
  --input_onnx_file_paths Constant_iou_threshold.onnx NonMaxSuppression11.onnx \
  --srcop_destop iou_threshold iou_threshold_var \
  --output_onnx_file_path NonMaxSuppression11.onnx

  snc4onnx \
  --input_onnx_file_paths Constant_score_threshold.onnx NonMaxSuppression11.onnx \
  --srcop_destop score_threshold score_threshold_var \
  --output_onnx_file_path NonMaxSuppression11.onnx


  ############ yxyx + NMS
  snc4onnx \
  --input_onnx_file_paths nms_yolact_edge_xyxy_yxyx.onnx NonMaxSuppression11.onnx \
  --srcop_destop boxes_xyxy_var boxes_var \
  --output_onnx_file_path NonMaxSuppression11.onnx

  sor4onnx \
  --input_onnx_file_path NonMaxSuppression11.onnx \
  --old_new "input_1" "boxes_var" \
  --output_onnx_file_path NonMaxSuppression11.onnx \
  --mode inputs
  ```
  ![image](https://user-images.githubusercontent.com/33194443/173185010-b71b92dd-be40-4c57-8abb-9dc418526a32.png)

  ```bash
  sit4onnx --input_onnx_file_path NonMaxSuppression11.onnx

  INFO: file: NonMaxSuppression11.onnx
  INFO: providers: ['TensorrtExecutionProvider', 'CPUExecutionProvider']
  INFO: input_name.1: boxes_var shape: [1, 19248, 4] dtype: float32
  INFO: input_name.2: scores_var shape: [1, 80, 19248] dtype: float32
  INFO: test_loop_count: 10
  INFO: total elapsed time:  79.51664924621582 ms
  INFO: avg elapsed time per pred:  7.951664924621581 ms
  INFO: output_name.1: selected_indices shape: [1600, 3] dtype: int64
  ```


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
  - `proto_out`: `float32 [N, 138, 138]`
    - `N` = The number of objects detected, filtered by NMS, and therefore less than 100.

![yolact_edge_mobilenetv2_550x550 onnx (2)](https://user-images.githubusercontent.com/33194443/172872136-0462bf2d-dd1e-45d0-abe0-e95293f7029f.png)

# Acknowledgments
https://github.com/yujin6056/yolactedge-onnx-conversion

