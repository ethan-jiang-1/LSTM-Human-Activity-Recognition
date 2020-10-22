import os

# two parameters can be used to select input and iter for saving model
# saving model are named as mode_save_{input}_{msstep} in its own subfolder
#
default_inputs = 6
default_model_save_iter = 500

default_sample_num = 128  # 128 is the original data

flags = []

flags.append("SAMPLE_NUM_32")
#flags.append("SAMPLE_NUM_64")
#flags.append("SAMPLE_NUM_128")
flags.append("LTSM_LAYER_1")
#flags.append("LTSM_LAYER_2")

def has_flag(flag):
    if flag in flags:
        return True
    return False


if has_flag("SAMPLE_NUM_32"):
    default_sample_num = 32
elif has_flag("SAMPLE_NUM_64"):
    default_sample_num = 64
elif has_flag("SAMPLE_NUM_128"):
    default_sample_num = 128
else:
    default_sample_num = 128


def update_env():
    print("update env, default_inputs: {} default_model_save_iter: {}".format(default_inputs, default_model_save_iter))
    os.environ['DATA_INPUTS_NUM'] = str(default_inputs)
    os.environ["DATA_MODEL_SAVE_STEP"] = str(default_model_save_iter)


update_env()


def alter_defaults(inputs, msstep):
    global default_inputs, default_model_save_iter
    default_inputs = inputs
    default_model_save_iter = msstep
    update_env()


print("*** Flags defined: ", flags)
print(" inputs (number of input vectors): ", default_inputs)
print(" model_save_iter (which step to choose convert/predit in the middle):", default_model_save_iter)
print("")