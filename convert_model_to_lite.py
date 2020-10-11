#!/usr/bin/env python
# encoding:utf8

# Version 1.0.0 (some previous versions are used in past commits)
from inspect import signature
from operator import truediv
import os
import traceback
import sys

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

using_v2 = False


def get_model_dir(inputs, step):
    return "model_save_{}_{}".format(inputs, step)


def get_model_path(inputs, step):
    return "model_tflite/model_save_{}_{}.tflite".format(inputs, step)


def find_converter(model_dir):

    if not using_v2:
        print("Using V1 Converter")
        from tensorflow.lite.python.lite import TFLiteConverter as tfc
        # from xt_tf.xa_lite import TFLiteConverter as tfc
        converter = tfc.from_saved_model("./" + model_dir)
        # converter.allow_custom_ops = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]

    else:
        print("Using V2 Converter")
        tf.enable_eager_execution()
        from tensorflow.lite.python.lite import TFLiteConverterV2 as tfc
        # from xt_tf.xa_lite import TFLiteConverterV2 as tfc
        tags = [tag_constants.SERVING]
        signature_keys = [
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        converter = tfc.from_saved_model(
            "./" + model_dir, tags=tags, signature_keys=signature_keys)

    return converter

# http://primo.ai/index.php?title=Converting_to_TensorFlow_Lite
# Converting a SavedModel.
def convert_and_save(model_dir, saved_model_path):
    print("try to convert {} to {} ".format(model_dir, saved_model_path))

    converter = find_converter(model_dir)
    tflite_model = converter.convert()

    with open(saved_model_path, "wb") as file:
        file.write(tflite_model)
        print("tflite saved in {}".format(saved_model_path))
        return True
    return False


def do_convert(inputs, msstep):
    dir_name = get_model_dir(inputs, msstep)
    saved_model_path = get_model_path(inputs, msstep)    
    print("received model need to be converted model {} to {} ".format(dir_name, saved_model_path))
    try:
        if not os.path.isdir(dir_name):
            print("\n** Error, no model folder found {}".format(dir_name))
            return False

        if not os.path.isdir("./model_tflite"):
            os.mkdir("./model_tflite")

        return convert_and_save(dir_name, saved_model_path)
    except Exception as ex:
        print("\n** Exception: {}".format(ex))
        traceback.print_exc()
        return False


if __name__ == '__main__':    # which model to load?  from model_save_XXX
    from s_defaults import default_inputs, default_msstep, alter_defaults
    if len(sys.argv) >= 3:
        inputs = int(sys.argv[1])
        msstep = int(sys.argv[2])
    else:
        inputs = default_inputs
        msstep = default_msstep
    alter_defaults()

    do_convert(inputs, msstep)
