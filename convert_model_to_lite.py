# Version 1.0.0 (some previous versions are used in past commits)
import os
import traceback
import sys 

# http://primo.ai/index.php?title=Converting_to_TensorFlow_Lite
# Converting a SavedModel.
def convert_and_save(model_name):
    print("try to convert {}".format(model_name))
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model("./" + model_name)
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()

    filename = "./model_tflite/" + model_name + ".tflite"
    with open(filename, "wb") as file:
        file.write(tflite_model)
        print("tflite saved in {}".format(filename))
        return True
    return False


def do_convert(name):
    print("received model need to be converted {}".format(name))
    try:
        model_name = "model_save_" + name
        if not os.path.isdir(model_name):
            print("\n** Error, no model folder found {}".format(model_name))
            return False

        if not os.path.isdir("./model_tflite"):
            os.mkdir("./model_tflite")

        return convert_and_save(model_name)
    except Exception as ex:
        print("\n** Exception: {}".format(ex))
        traceback.print_exc()
        return False


if __name__ == '__main__':    # which model to load?  from model_save_XXX
    name = "500"

    if len(sys.argv) >= 2:
        name = sys.argv[1]

    do_convert(name)
