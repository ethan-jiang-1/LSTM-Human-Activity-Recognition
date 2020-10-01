# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf

# Converting a SavedModel.
converter = tf.lite.TFLiteConverter.from_saved_model("./model_keep/xltsm_model_final")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
