import copy
from typing import Tuple
import cv2
import argparse
import onnxruntime
import numpy as np
import time


LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

COLORS = [
    [244,  67,  54],
    [233,  30,  99],
    [156,  39, 176],
    [103,  58, 183],
    [ 63,  81, 181],
    [ 33, 150, 243],
    [  3, 169, 244],
    [  0, 188, 212],
    [  0, 150, 136],
    [ 76, 175,  80],
    [139, 195,  74],
    [205, 220,  57],
    [255, 235,  59],
    [255, 193,   7],
    [255, 152,   0],
    [255,  87,  34],
    [121,  85,  72],
    [158, 158, 158],
    [ 96, 125, 139],
]


def get_color(idx):
    color_idx = idx % len(COLORS)
    color = COLORS[color_idx]
    return color


def sanitize_coordinates(
    _x1: np.ndarray,
    _x2: np.ndarray,
    img_size: int,
    padding: int=0,
) -> Tuple[np.ndarray, np.ndarray]:

    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    _x1 = _x1.astype(np.int32)
    _x2 = _x2.astype(np.int32)
    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.where((x1-padding) > 0, (x1-padding), 0)
    x2 = np.where((x2+padding) < img_size, (x2+padding), img_size)
    return x1, x2


def crop(masks: np.ndarray, boxes:np.ndarray, padding:int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [n, h, w] tensor of masks
        - boxes should be a size [n, 6] tensor of bbox coords in relative point form
    """
    if len(masks.shape) < 3:
        masks = masks[np.newaxis, ...]
    n, h, w = masks.shape
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)
    rows = np.arange(w, dtype=x1.dtype).reshape(1, 1, -1)
    rows = np.broadcast_to(rows, (n, h, w))
    cols = np.arange(h, dtype=x1.dtype).reshape(1, 1, -1)
    cols = np.broadcast_to(cols, (n, h, w))
    masks_left  = rows >= x1.reshape(-1, 1, 1)
    masks_right = rows <  x2.reshape(-1, 1, 1)
    masks_up    = cols >= y1.reshape(-1, 1, 1)
    masks_down  = cols <  y2.reshape(-1, 1, 1)
    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask


def main(args):

    # Load Face Detection Model
    face_detection_model = 'yolact_edge_mobilenetv2_550x550.onnx'
    session_option_detection = onnxruntime.SessionOptions()
    session_option_detection.log_severity_level = 3
    sess = onnxruntime.InferenceSession(
        face_detection_model,
        sess_options=session_option_detection,
        providers=[
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )
    input_name = sess.get_inputs()[0].name
    input_shapes = sess.get_inputs()[0].shape
    # output_names = [output.name for output in sess.get_outputs()]

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

        start = time.time()
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
        masks = results[1][0]
        masks = crop(
            masks,
            x1y1x2y2_scores_classes_4x1x1_result,
        )

        for x1y1x2y2_scores_classes_4x1x1 in x1y1x2y2_scores_classes_4x1x1_result:
            score = x1y1x2y2_scores_classes_4x1x1[4]
            classid = int(x1y1x2y2_scores_classes_4x1x1[5])
            if score > 0.60:
                x_min = int(x1y1x2y2_scores_classes_4x1x1[0] * cap_width)
                y_min = int(x1y1x2y2_scores_classes_4x1x1[1] * cap_height)
                x_max = int(x1y1x2y2_scores_classes_4x1x1[2] * cap_width)
                y_max = int(x1y1x2y2_scores_classes_4x1x1[3] * cap_height)

                cv2.putText(
                    canvas,
                    f'{LABELS[classid]} {score:.2f}',
                    (x_min, (y_min-5) if (y_min-5) > 20 else 20),
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

        cv2.putText(
            canvas,
            f'{(time.time()-start)*1000:.2f}ms, {1000/(time.time()-start)/1000:.2f}FPS',
            (10,30),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

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
