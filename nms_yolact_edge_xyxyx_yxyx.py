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

x1 = boxes[...,0][...,np.newaxis]
y1 = boxes[...,1][...,np.newaxis]
x2 = boxes[...,2][...,np.newaxis]
y2 = boxes[...,3][...,np.newaxis]
boxes_y1x1y2x2 = tf.concat([y1,x1,y2,x2], axis=2)

model = tf.keras.models.Model(inputs=[boxes], outputs=[boxes_y1x1y2x2])
model.summary()
output_path = 'saved_model_postprocess'
tf.saved_model.save(model, output_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
open(f"{output_path}/nms_yolact_edge_xyxy_yxyx.tflite", "wb").write(tflite_model)