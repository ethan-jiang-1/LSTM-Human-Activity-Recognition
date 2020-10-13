import os

default_inputs = 6
default_msstep = 500
default_ssample = 128 # 128 is the original data

def update_env():
    print("update env, default_inputs: {} default_msstep: {}".format(default_inputs, default_msstep))
    os.environ['DATA_INPUTS_NUM'] = str(default_inputs)
    os.environ["DATA_MODEL_SAVE_STEP"] = str(default_msstep)
    os.environ["DATA_MODEL_SAMPLE"] = str(default_ssample)


update_env()


def alter_defaults(inputs, msstep):
    global default_inputs, default_msstep
    default_inputs = inputs
    default_msstep = msstep
    update_env()
