
import os

default_inputs = 6
default_msstep = 500

def update_env():
    print("update env, default_inputs: {} default_msstep: {}".format(default_inputs, default_msstep))
    os.environ['DATA_INPUTS_NUM'] = str(default_inputs)
    os.environ["DATA_MODEL_SAVE_STEP"] = str(default_msstep)


update_env()


def alter_defaults(inputs, msstep):
    global default_inputs, default_msstep
    default_inputs = inputs
    default_msstep = msstep
    update_env()
