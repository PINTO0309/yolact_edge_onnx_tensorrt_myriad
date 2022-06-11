import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
from pprint import pprint
np.random.seed(0)


# Create a model
selected_indices = tf.keras.layers.Input(
    shape=[
        3,
    ],
    dtype=tf.int64,
)

classes = tf.cast(selected_indices[:,1], dtype=tf.float32)[np.newaxis,:,np.newaxis]
box_indexes = selected_indices[:,2]

model = tf.keras.models.Model(inputs=[selected_indices], outputs=[classes,box_indexes])
model.summary()
output_path = 'saved_model_postprocess'
tf.saved_model.save(model, output_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
open(f"{output_path}/nms_yolact_edge_score_indices.tflite", "wb").write(tflite_model)