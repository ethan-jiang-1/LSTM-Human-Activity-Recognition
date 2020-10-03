# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import os
import traceback 

# http://primo.ai/index.php?title=Converting_to_TensorFlow_Lite
# Converting a SavedModel.
# converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()


# Converting a SavedModel.
def convert_and_save(model_name):
    print("try to convert {}".format(model_name))
    converter = tf.lite.TFLiteConverter.from_saved_model("./" + model_name)
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()

    filename = "./tflite/" + model_name + ".tflite"
    with open(filename, "wb") as file:
        file.write(tflite_model)
        print("tflite saved in {}".format(filename))
        return True
    return False

try:
    name = "500"
    model_name = "model_save_" + name

    if not os.path.isdir("./tflite"):
        os.mkdir("./tflite")

    convert_and_save(model_name)
except Exception as ex:
    print("\n** Exception: {}".format(ex))
    traceback.print_exc()

