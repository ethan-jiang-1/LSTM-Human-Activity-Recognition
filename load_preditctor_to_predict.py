#!/usr/bin/env python
# encoding:utf8

import tensorflow as tf
import os
import traceback 
import sys

from xt_tf.xa_saved_model_predictor import SavedModelPredictor
import numpy as np
from sklearn import metrics
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_data_loader import load_all
from s_graph import inspect_graph

# load dataset from data_loader
dh = load_all()
#X_train = dh.X_train
#y_train = dh.y_train
X_test = dh.X_test
y_test = dh.y_test
LABELS = dh.LABELS


def do_load_predictor(name):
    print("received model need to be converted {}".format(name))
    try:
        model_name = "model_save_" + name
        if not os.path.isdir(model_name):
            print("\n** Error, no model folder found {}".format(model_name))
            return False

        dir_name = "./" + model_name
        smp = SavedModelPredictor(dir_name)
        return smp
    except Exception as ex:
        prompt_red("\n** Exception: {}".format(ex))
        traceback.print_exc()
        return None


n_classes = 6
n_steps = len(X_test[0])  # 128 timesteps per series
n_input = len(X_test[0][0])  # 9 input parameters per timestep


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape, dtype=np.float32)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s

def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    # Returns FLOATS
    return np.eye(n_classes, dtype=np.int32)[np.array(y_, dtype=np.int32)]


def do_predict_test_set_all(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_set_all", graph=smp.graph)

    sess = smp.session

    # Accuracy for test data
    x = sess.graph.get_tensor_by_name('Input/my_x_input:0')
    pred = sess.graph.get_tensor_by_name('Output/my_pred:0')
    one_hot_predictions = sess.run(
        pred,
        feed_dict={
            x: X_test
        }
    )
    predictions = one_hot_predictions.argmax(1)

    print("Precision: {:.4f}%".format(
        100 * metrics.precision_score(y_test, predictions, average="weighted")))
    print("Recall: {:.4f}%".format(
        100 * metrics.recall_score(y_test, predictions, average="weighted")))
    print("f1_score: {:.4f}%".format(
        100 * metrics.f1_score(y_test, predictions, average="weighted")))

    return True


def do_predict_test_set_skip_A(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_set_skip_A", graph=smp.graph)

    sess = smp.session

    skip_ratio = 10

    # Accuracy for test data
    x = sess.graph.get_tensor_by_name('Input/my_x_input:0')
    pred = sess.graph.get_tensor_by_name('Output/my_pred:0')
    one_hot_predictions = sess.run(
        pred,
        feed_dict={
            x: X_test[::skip_ratio],
        }
    )
    predictions = one_hot_predictions.argmax(1)

    print("Precision: {:.4f}%".format(
        100 * metrics.precision_score(y_test[::skip_ratio], predictions, average="weighted")))
    print("Recall: {:.4f}%".format(
        100 * metrics.recall_score(y_test[::skip_ratio], predictions, average="weighted")))
    print("f1_score: {:.4f}%".format(
        100 * metrics.f1_score(y_test[::skip_ratio], predictions, average="weighted")))

    return True


def do_predict_test_set_skip_B(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_set_skip_B", graph=smp.graph)

    sess = smp.session

    skip_ratio = 100

    # Accuracy for test data
    x = sess.graph.get_tensor_by_name('Input/my_x_input:0')
    pred = sess.graph.get_tensor_by_name('Output/my_pred:0')
    one_hot_predictions = sess.run(
        pred,
        feed_dict={
            x: X_test[::skip_ratio],
        }
    )
    predictions = one_hot_predictions.argmax(1)

    print("Precision: {:.4f}%".format(
        100 * metrics.precision_score(y_test[::skip_ratio], predictions, average="weighted")))
    print("Recall: {:.4f}%".format(
        100 * metrics.recall_score(y_test[::skip_ratio], predictions, average="weighted")))
    print("f1_score: {:.4f}%".format(
        100 * metrics.f1_score(y_test[::skip_ratio], predictions, average="weighted")))

    return True


def do_predict_test_set_one_C(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_set_one_C", graph=smp.graph)

    sess = smp.session

    # Accuracy for test data
    x = sess.graph.get_tensor_by_name('Input/my_x_input:0')
    pred = sess.graph.get_tensor_by_name('Output/my_pred:0')

    matched = 0
    unmatched = 0
    for cn in range(0, len(y_test)):
        one_hot_predictions = sess.run(
            pred,
            feed_dict={
                x: X_test[cn:cn+1],
            }
        )
        pred_v = one_hot_predictions.argmax(1)
        real_v = y_test[cn:cn+1]
        # print(pred_v, real_v)
        if pred_v[0] == real_v[0][0]:
            matched += 1
        else:
            unmatched += 1
    print("accuracy one by one: {:4f} on {}/{}".format((float(matched)/float(matched+unmatched)), matched, unmatched))
    return True


def do_predict_test_one_X(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_one_X", graph=smp.graph)

    matched = 0
    unmatched = 0
    for cn in range(0, len(y_test)):
        input_one = X_test[cn:cn+1]
        outputs = smp({"x": input_one})
        pred_v = outputs["y"].argmax(1)
        real_v = y_test[cn:cn+1]

        # print(pred_v, real_v)
        if pred_v[0] == real_v[0][0]:
            matched += 1
        else:
            unmatched += 1
    print("accuracy one by one: {:4f} on {}/{}".format(
        (float(matched) / float(matched + unmatched)), matched, unmatched))
    return True


if __name__ == '__main__':    # which model to load?  from model_save_XXX
    name = "500"

    if len(sys.argv) >= 2:
        name = sys.argv[1]

    smp = do_load_predictor(name)
    if smp is not None:
        do_predict_test_set_all(smp)
        do_predict_test_set_skip_A(smp)
        do_predict_test_set_skip_B(smp)
        do_predict_test_set_one_C(smp)
        do_predict_test_one_X(smp)
