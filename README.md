# yolact_edge_onnx_tensorrt_myriad
Provides a conversion flow for **`YOLACT_Edge`** to models compatible with ONNX, TensorRT and Myriad (OAK). My own implementation of post-processing allows for e2e inference.

# Official Repo
https://github.com/haotian-liu/yolact_edge

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

# Acknowledgments
https://github.com/yujin6056/yolactedge-onnx-conversion
