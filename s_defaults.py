import os

default_inputs = 6
default_msstep = 500
default_ssample = 32 #  128 is the original data

flags = []
#flags.append("SAMPLE_NUM_32")
flags.append("SAMPLE_NUM_64")
# flags.append("LTSM_LAYER_1")
flags.append("LTSM_LAYER_2")
def has_flag(flag):
    if flag in flags:
        return True
    return False


if has_flag("SAMPLE_NUM_32"):
    default_ssample = 32
elif has_flag("SAMPLE_NUM_64"):
    default_ssample = 64
else:
    default_ssample = 128


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

print("*** Flags defined: ", flags)
print(" inputs (number of input vectors): ", default_inputs)
print(" mssteps (which step to choose convert/predit in the middle):", default_msstep)
print("")