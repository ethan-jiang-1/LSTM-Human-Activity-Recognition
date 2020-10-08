import tensorflow as tf
import os
import traceback 
import sys

import numpy as np
from sklearn import metrics
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_data_loader import load_all
from s_graph import inspect_graph

# load dataset from data_loader
dh = load_all()
X_train = dh.X_train
X_test = dh.X_test
y_train = dh.y_train
y_test = dh.y_test
LABELS = dh.LABELS

def do_load_model(name):
    print("received model need to be converted {}".format(name))
    try:
        model_name = "model_save_" + name
        if not os.path.isdir(model_name):
            print("\n** Error, no model folder found {}".format(model_name))
            return False

        dir_name = "./" + model_name
        # import pdb; pdb.set_trace()
        
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


n_classes = 6
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep

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
    y = sess.graph.get_tensor_by_name('Output/my_y_output:0')
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
        y = sess.graph.get_tensor_by_name('Output/my_y_output:0')
        pred = sess.graph.get_tensor_by_name('Model/my_pred:0')
        accuracy = sess.graph.get_tensor_by_name('Accuray/my_accuracy:0')
        print(x, y, pred, accuracy)

        op_x = sess.graph.get_operation_by_name('Input/my_x_input')
        op_y = sess.graph.get_operation_by_name('Output/my_y_output')
        op_pred = sess.graph.get_operation_by_name('Model/my_pred')
        op_accuracy = sess.graph.get_operation_by_name('Accuray/my_accuracy')
        print(op_x, op_y, op_pred, op_accuracy)

        ctx = sess.graph.get_tensor_by_name("my_c_input:0")
        cty = sess.graph.get_tensor_by_name("my_c_output:0")
        op_ctx = sess.graph.get_operation_by_name("my_c_input")
        op_cty = sess.graph.get_operation_by_name("my_c_output")
        print(ctx, cty, op_ctx, op_cty)

        nc = sess.graph.get_operation_by_name("my_n_classes")
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

        tensor_info_outputs = sdef.outputs
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
    name = "100"

    if len(sys.argv) >= 2:
        name = sys.argv[1]

    sess, meta_info_def = do_load_model(name)
    if check_graph(sess, meta_info_def):
        check_signature_def(sess, meta_info_def)
        if sess is not None:
            do_predict_test_set(sess, meta_info_def)
