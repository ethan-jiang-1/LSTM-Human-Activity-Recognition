import tensorflow as tf
import os
import traceback 
import sys


model = None
sess = None


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

        tags = []

        model = tf.saved_model.load(sess, tags, dir_name)
        print(model)
        return True

    except Exception as ex:
        print("\n** Exception: {}".format(ex))
        traceback.print_exc()
        return False


def do_predict_test_set():
    if model is None:
        return False
    return True


if __name__ == '__main__':    # which model to load?  from model_save_XXX
    name = "500"

    if len(sys.argv) >= 2:
        name = sys.argv[1]

    do_load_model(name)
    if model is not None:
        do_predict_test_set()
