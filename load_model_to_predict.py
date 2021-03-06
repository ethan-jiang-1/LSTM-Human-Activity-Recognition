#!/usr/bin/env python
# encoding:utf8

from s_defaults import default_inputs, default_model_save_iter, alter_defaults
import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


import tensorflow as tf
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
import traceback 
import sys
import logging
import numpy as np
from sklearn import metrics
from s_data_loader import load_all, find_inputs_num
from s_graph import inspect_graph

tf.get_logger().setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
logging.getLogger('tensorflow').disabled = True
logging.getLogger('tensorflow.core').disabled = True


# load dataset from data_loader
X_train = None
X_test = None
y_train = None
y_test = None
LABELS = None

n_classes = 6
n_steps = 128
n_input = default_inputs

def load_data_by_inputs(inputs):
    global X_train, X_test, y_train, y_test, LABELS
    global n_steps, n_input 
    os.environ['DATA_INPUTS_NUM'] = str(inputs)
    dh = load_all()
    X_train = dh.X_train
    X_test = dh.X_test
    y_train = dh.y_train
    y_test = dh.y_test
    LABELS = dh.LABELS

    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep

def get_model_dir(inputs, step):
    return "model_save_{}_{}".format(inputs, step)


def do_load_model(model_dir):
    print("received model need to be converted {}".format(model_dir))
    try:
        if not os.path.isdir(model_dir):
            print("\n** Error, no model folder found {}".format(model_dir))
            return False

        dir_name = "./" + model_dir
        
        sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True, device_count={'GPU': 0}))
        init = tf.global_variables_initializer()
        sess.run(init)  

        tags = [tf.saved_model.tag_constants.SERVING]

        prompt_yellow("load saved_model in current sess...")
        meta_info_def = tf.saved_model.load(sess, tags, dir_name)
        #print(meta_info_def)
        prompt_yellow("model loaded")
        inspect_graph("inspect_loaded_model")
        return sess, meta_info_def

    except Exception as ex:
        prompt_red("\n** Exception: {}".format(ex))
        traceback.print_exc()
        return None, None


def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    # Returns FLOATS
    return np.eye(n_classes, dtype=np.int32)[np.array(y_, dtype=np.int32)]


def do_predict_test_set(sess, meta_info_def):
    if meta_info_def is None:
        return False
    
    # Accuracy for test data
    y_test_oh = one_hot(y_test)
    x = sess.graph.get_tensor_by_name('Input/my_x_input:0')
    y = sess.graph.get_tensor_by_name('Input/my_y_input:0')
    pred = sess.graph.get_tensor_by_name('Output/my_pred:0')
    accuracy = sess.graph.get_tensor_by_name('Accuray/my_accuracy:0')
    print(x, y, pred, accuracy)
    one_hot_predictions, final_accuracy, = sess.run(
        [pred, accuracy],
        feed_dict={
            x: X_test,
            y: y_test_oh        
        }
    )
    predictions = one_hot_predictions.argmax(1)

    print("Testing Accuracy: {}%".format(100 * final_accuracy))
    print("")
    print("Precision: {}%".format(
        100 * metrics.precision_score(y_test, predictions, average="weighted")))
    print("Recall: {}%".format(
        100 * metrics.recall_score(y_test, predictions, average="weighted")))
    print("f1_score: {}%".format(
        100 * metrics.f1_score(y_test, predictions, average="weighted")))

    return True


def check_graph(sess, meta_graph_def):
    try:
        inspect_graph("check_graph")

        x = sess.graph.get_tensor_by_name('Input/my_x_input:0')
        y = sess.graph.get_tensor_by_name('Input/my_y_input:0')
        pred = sess.graph.get_tensor_by_name('Output/my_pred:0')
        accuracy = sess.graph.get_tensor_by_name('Accuray/my_accuracy:0')
        print(x, y, pred, accuracy)

        op_x = sess.graph.get_operation_by_name('Input/my_x_input')
        op_y = sess.graph.get_operation_by_name('Input/my_y_input')
        op_pred = sess.graph.get_operation_by_name('Output/my_pred')
        op_accuracy = sess.graph.get_operation_by_name('Accuray/my_accuracy')
        print(op_x, op_y, op_pred, op_accuracy)

        nc = sess.graph.get_operation_by_name("my_cn_classes")
        print(nc)

        return True
    except Exception as ex:
        prompt_red("\n** Exception: {}".format(ex))
        traceback.print_exc()
    return False


def check_signature_def(sess, meta_graph_def):
    from tensorflow.python.saved_model import signature_def_utils
    from tensorflow.python.saved_model import signature_constants

    try:
        op_signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        print(op_signature_key)
        import_scope = None

        if op_signature_key not in meta_graph_def.signature_def:
            return None

        print(meta_graph_def.signature_def)
        sdef = meta_graph_def.signature_def[op_signature_key]
        prompt_blue(sdef)

        # tensor_info_outputs = sdef.outputs
        op_outputs_signature_key = "y"

        op_sdef = signature_def_utils.load_op_from_signature_def(sdef, 
                                                                 op_outputs_signature_key,
                                                                 import_scope=import_scope)
        prompt_green(op_sdef)
        return op_sdef
    except Exception as ex:
        prompt_red("\n** Exception: {}".format(ex))
        traceback.print_exc()
    return None


if __name__ == '__main__':    # which model to load?  from model_save_XXX
    if len(sys.argv) >= 3:
        inputs = int(sys.argv[1])
        msstep = int(sys.argv[2])
    else:
        inputs = default_inputs
        msstep = default_model_save_iter

    alter_defaults(inputs, msstep)
    
    load_data_by_inputs(inputs)
    model_dir = get_model_dir(inputs, msstep)

    sess, meta_info_def = do_load_model(model_dir)
    if check_graph(sess, meta_info_def):
        check_signature_def(sess, meta_info_def)
        if sess is not None:
            do_predict_test_set(sess, meta_info_def)
