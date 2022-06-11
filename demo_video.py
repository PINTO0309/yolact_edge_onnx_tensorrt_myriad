import copy
import cv2
import argparse
import onnxruntime
import numpy as np


COLORS = [
    [[[[244,  67,  54]]]],
    [[[[233,  30,  99]]]],
    [[[[156,  39, 176]]]],
    [[[[103,  58, 183]]]],
    [[[[ 63,  81, 181]]]],
    [[[[ 33, 150, 243]]]],
    [[[[  3, 169, 244]]]],
    [[[[  0, 188, 212]]]],
    [[[[  0, 150, 136]]]],
    [[[[ 76, 175,  80]]]],
    [[[[139, 195,  74]]]],
    [[[[205, 220,  57]]]],
    [[[[255, 235,  59]]]],
    [[[[255, 193,   7]]]],
    [[[[255, 152,   0]]]],
    [[[[255,  87,  34]]]],
    [[[[121,  85,  72]]]],
    [[[[158, 158, 158]]]],
    [[[[ 96, 125, 139]]]],
]


def get_color(idx):
    color_idx = idx % len(COLORS)
    color = COLORS[color_idx]
    return color


def main(args):

    # Load Face Detection Model
    face_detection_model = 'yolact_edge_mobilenetv2_550x550.onnx'
    session_option_detection = onnxruntime.SessionOptions()
    session_option_detection.log_severity_level = 3
    sess = onnxruntime.InferenceSession(
        face_detection_model,
        sess_options=session_option_detection,
        providers=[
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    input_name = sess.get_inputs()[0].name
    input_shapes = sess.get_inputs()[0].shape
    output_names = [output.name for output in sess.get_outputs()]

    cap_width = int(args.height_width.split('x')[1])
    cap_height = int(args.height_width.split('x')[0])
    if args.device.isdecimal():
        cap = cv2.VideoCapture(int(args.device))
    else:
        cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    model_input_h = int(input_shapes[2])
    model_input_w = int(input_shapes[3])

    WINDOWS_NAME = 'Demo'
    cv2.namedWindow(WINDOWS_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWS_NAME, cap_width, cap_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        canvas = copy.deepcopy(frame)

        # Resize
        resized_frame = cv2.resize(frame, (model_input_w, model_input_h))
        width = resized_frame.shape[1]
        height = resized_frame.shape[0]
        # BGR to RGB
        rgb = resized_frame[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # normalize to [0, 1] interval
        # chw = np.asarray(chw / 255., dtype=np.float32)
        # hwc --> nhwc
        nchw = chw[np.newaxis, ...]
        # Uint8 -> Float32
        nchw = nchw.astype(np.float32)

        results = sess.run(
            None,
            {input_name: nchw}
        )

        x1y1x2y2_scores_classes_4x1x1_result = results[0][0]
        # x1y1x2y2_scores_classes_4x1x1_result = results[0]

        # proto_out = results[1]
        # detection_count = len(x1y1x2y2_scores_classes_4x1x1_result)

        for x1y1x2y2_scores_classes_4x1x1 in x1y1x2y2_scores_classes_4x1x1_result:
            score = x1y1x2y2_scores_classes_4x1x1[4]
            if score > 0.60:
                x_min = int(x1y1x2y2_scores_classes_4x1x1[0] * cap_width)
                y_min = int(x1y1x2y2_scores_classes_4x1x1[1] * cap_height)
                x_max = int(x1y1x2y2_scores_classes_4x1x1[2] * cap_width)
                y_max = int(x1y1x2y2_scores_classes_4x1x1[3] * cap_height)

                cv2.putText(
                    canvas,
                    f'{score:.2f}',
                    (x_min, y_min if y_min > 20 else 20),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    canvas,
                    (x_min, y_min),
                    (x_max, y_max),
                    color=(255, 0, 0),
                    thickness=2
                )

                # masks = proto_out[..., None]
                # colors = np.asarray(
                #     [get_color(idx) for idx in range(detection_count)],
                #     dtype=np.int32,
                # )
                # masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

                # full_masks = np.zeros(cap_height, cap_width)
                # mask_w = x_max - x_min
                # mask_h = y_max - y_min
                # if mask_w * mask_h <= 0 or mask_w < 0:
                #     continue
                # mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
                # mask = mask.gt(0.5).float()
                # full_masks[jdx, y1:y2, x1:x2] = mask


        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        cv2.imshow(WINDOWS_NAME, canvas)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default='0',
        help='Path of the mp4 file or device number of the USB camera. Default: 0',
    )
    parser.add_argument(
        "--height_width",
        type=str,
        default='480x640',
        help='{H}x{W}. Default: 480x640',
    )
    args = parser.parse_args()
    main(args)
