#!/usr/bin/env python
# encoding:utf8

import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['DATA_INPUTS_NUM'] = '9'

import sys
from tensorflow.lite.python.lite import Interpreter
# import numpy as np
# import tensorflow as tf

from s_data_loader import load_all, find_inputs_num


def get_model_path(inputs, step):
    return "model_tflite/model_save_{}_{}.tflite".format(inputs, step)


def do_load_lite_predictor(model_path):
    print("using tlite model from {} to predict".format(model_path))

    # load dataset from data_loader
    dh = load_all()
    #X_train = dh.X_train
    #y_train = dh.y_train
    X_test = dh.X_test
    y_test = dh.y_test
    #LABELS = dh.LABELS

    # Load TFLite model and allocate tensors.
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(
    #    input_shape), dtype=np.float32)

    matched = 0
    unmatched = 0
    for cn in range(0, len(y_test)):
        input_data = X_test[cn:cn + 1]
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print(output_data)
        pred_v = output_data.argmax(1)
        real_v = y_test[cn:cn + 1]
        # print(pred_v, real_v)
        if pred_v[0] == real_v[0][0]:
            matched += 1
        else:
            unmatched += 1
    print("accuracy one by one: {:4f} on {}/{}".format(
        (float(matched) / float(matched + unmatched)), matched, unmatched))
    return True


if __name__ == '__main__':    # which model to load?  from model_save_XXX
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    else:
        inputs = find_inputs_num()
        step = 100
        model_path = get_model_path(inputs, step)

    do_load_lite_predictor(model_path)
