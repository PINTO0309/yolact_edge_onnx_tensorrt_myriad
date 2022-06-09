import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
from pprint import pprint
np.random.seed(0)


BOXES=19248

# Create a model
boxes = tf.keras.layers.Input(
    shape=[
        BOXES,
        4,
    ],
    batch_size=1,
    dtype=tf.float32,
)

scores = tf.keras.layers.Input(
    shape=[
        BOXES,
    ],
    batch_size=1,
    dtype=tf.float32,
)

classes = tf.keras.layers.Input(
    shape=[
        BOXES,
    ],
    batch_size=1,
    dtype=tf.int64,
)

masks = tf.keras.layers.Input(
    shape=[
        BOXES,
        32,
    ],
    batch_size=1,
    dtype=tf.float32,
)


boxes_non_batch = tf.squeeze(boxes)
x1 = boxes_non_batch[:,0][:,np.newaxis]
y1 = boxes_non_batch[:,1][:,np.newaxis]
x2 = boxes_non_batch[:,2][:,np.newaxis]
y2 = boxes_non_batch[:,3][:,np.newaxis]
boxes_y1x1y2x2 = tf.concat([y1,x1,y2,x2], axis=1)

scores_non_batch = tf.squeeze(scores)
classes_non_batch = tf.squeeze(classes)
masks_non_batch = tf.squeeze(masks)

selected_indices = tf.cast(tf.image.non_max_suppression(
    boxes=boxes_y1x1y2x2,
    scores=scores_non_batch,
    max_output_size=100,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
), dtype=tf.int32)

selected_boxes =  tf.gather(
    boxes_non_batch,
    selected_indices
)

selected_scores = tf.gather(
    scores_non_batch,
    selected_indices
)
selected_scores = tf.expand_dims(selected_scores, axis=1)

selected_classes = tf.cast(tf.gather(
    classes_non_batch,
    selected_indices
), dtype=tf.float32)
selected_classes = tf.expand_dims(selected_classes, axis=1)

selected_masks = tf.gather(
    masks_non_batch,
    selected_indices
)


outputs = tf.concat([selected_boxes, selected_scores, selected_classes, selected_masks], axis=1)

model = tf.keras.models.Model(inputs=[boxes,scores,classes,masks], outputs=[outputs])
model.summary()
output_path = 'saved_model_postprocess'
tf.saved_model.save(model, output_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
open(f"{output_path}/nms_yolact_edge.tflite", "wb").write(tflite_model)