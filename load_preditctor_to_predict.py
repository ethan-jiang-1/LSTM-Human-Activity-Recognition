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


def do_predict_test_set(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_set", graph=smp.graph)

    sess = smp.session

    # Accuracy for test data
    y_test_oh = one_hot(y_test)
    x = sess.graph.get_tensor_by_name('my_x_input:0')
    y = sess.graph.get_tensor_by_name('my_y_output:0')
    pred = sess.graph.get_tensor_by_name('Model/my_pred:0')
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

    print("Testing Accuracy: {}% over {}".format(
        100 * final_accuracy, len(y_test_oh)))
    print("")
    print("Precision: {}%".format(
        100 * metrics.precision_score(y_test, predictions, average="weighted")))
    print("Recall: {}%".format(
        100 * metrics.recall_score(y_test, predictions, average="weighted")))
    print("f1_score: {}%".format(
        100 * metrics.f1_score(y_test, predictions, average="weighted")))

    return True


def do_predict_test_set_skip_A(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_set_skip_A", graph=smp.graph)

    sess = smp.session

    skip_ratio = 10

    # Accuracy for test data
    y_test_oh = one_hot(y_test[::skip_ratio])
    x = sess.graph.get_tensor_by_name('my_x_input:0')
    y = sess.graph.get_tensor_by_name('my_y_output:0')
    pred = sess.graph.get_tensor_by_name('Model/my_pred:0')
    accuracy = sess.graph.get_tensor_by_name('Accuray/my_accuracy:0')
    print(x, y, pred, accuracy)
    one_hot_predictions, final_accuracy, = sess.run(
        [pred, accuracy],
        feed_dict={
            x: X_test[::skip_ratio],
            y: y_test_oh
        }
    )
    predictions = one_hot_predictions.argmax(1)

    print("Testing Accuracy: {}% over {} data".format(100 * final_accuracy, len(y_test_oh)))
    print("")
    print("Precision: {}%".format(
        100 * metrics.precision_score(y_test[::skip_ratio], predictions, average="weighted")))
    print("Recall: {}%".format(
        100 * metrics.recall_score(y_test[::skip_ratio], predictions, average="weighted")))
    print("f1_score: {}%".format(
        100 * metrics.f1_score(y_test[::skip_ratio], predictions, average="weighted")))

    return True


def do_predict_test_set_skip_B(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_set_skip_B", graph=smp.graph)

    sess = smp.session

    skip_ratio = 100

    # Accuracy for test data
    y_test_oh = one_hot(y_test[::skip_ratio])
    x = sess.graph.get_tensor_by_name('my_x_input:0')
    y = sess.graph.get_tensor_by_name('my_y_output:0')
    pred = sess.graph.get_tensor_by_name('Model/my_pred:0')
    accuracy = sess.graph.get_tensor_by_name('Accuray/my_accuracy:0')
    print(x, y, pred, accuracy)
    one_hot_predictions, final_accuracy, = sess.run(
        [pred, accuracy],
        feed_dict={
            x: X_test[::skip_ratio],
            y: y_test_oh
        }
    )
    predictions = one_hot_predictions.argmax(1)

    print("Testing Accuracy: {}% over {} data".format(100 * final_accuracy, len(y_test_oh)))
    print("")
    print("Precision: {}%".format(
        100 * metrics.precision_score(y_test[::skip_ratio], predictions, average="weighted")))
    print("Recall: {}%".format(
        100 * metrics.recall_score(y_test[::skip_ratio], predictions, average="weighted")))
    print("f1_score: {}%".format(
        100 * metrics.f1_score(y_test[::skip_ratio], predictions, average="weighted")))

    return True


def do_predict_test_one_X(smp):
    if smp is None:
        return False

    inspect_graph("do_predict_test_one_X", graph=smp.graph)

    matched = 0
    not_machted = 0
    for step in range(0, 100):
        #prompt_yellow("predict {}".format(step))
        # po_batch_one_xs = extract_batch_size(X_test, step, 1)
        po_batch_one_xs = X_test[step:step+1]
        #print("X_test", po_batch_one_xs.shape, po_batch_one_xs)
        pred = smp({"x": po_batch_one_xs})
        po_one_hot_predictions = pred["y"]
        p_max = po_one_hot_predictions.argmax(1)
        #prompt_yellow("pred", po_one_hot_predictions,
        #            po_one_hot_predictions.argmax(1))

        #po_batch_one_ys = extract_batch_size(y_test, step, 1)
        po_batch_one_ys = y_test[step:step+1]
        #print("y_test", po_batch_one_ys.shape, po_batch_one_ys)
        po_batch_one_ys_oh = one_hot(po_batch_one_ys)
        r_max = po_batch_one_ys_oh.argmax(1)
        #prompt_yellow("real", po_batch_one_ys_oh, po_batch_one_ys_oh.argmax(1))
        if p_max == r_max:
            prompt_green("{} predict and actual are matched: {} {}".format(step, p_max, r_max))
            matched += 1
        else:
            prompt_red("{} predict and actual not matched: {} {}".format(step, p_max, r_max))
            not_machted += 1
    prompt_yellow("matched/mot_matched: {}/{}".format(matched, not_machted))


if __name__ == '__main__':    # which model to load?  from model_save_XXX
    name = "9999"

    if len(sys.argv) >= 2:
        name = sys.argv[1]

    smp = do_load_predictor(name)
    if smp is not None:
        do_predict_test_set(smp)
        do_predict_test_set_skip_A(smp)
        do_predict_test_set_skip_B(smp)
        do_predict_test_one_X(smp)
