# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf

# http://primo.ai/index.php?title=Converting_to_TensorFlow_Lite
# Converting a SavedModel.
# converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()

# Converting a SavedModel.
converter = tf.lite.TFLiteConverter.from_saved_model("./model_save_400")
converter.allow_custom_ops = True
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
