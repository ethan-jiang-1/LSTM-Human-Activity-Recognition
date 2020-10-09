# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor
from s_graph import inspect_graph

save_ses_enabled = False


def save_train_board_ses_pb(ses, step):
    try:
        tf.train.write_graph(
            ses.graph_def, '', './model_save_ses/model_save-{}.pb'.format(step), as_text=False)
    except Exception as ex:
        prompt_red("**model_ses {} failed to saved {}".format(step, ex))
        with ConsoleColor(ConsoleColor.RED):
            traceback.print_exc()


def save_model_ses(ses, step):
    if not save_ses_enabled:
        return False

    info = inspect_graph("saved_model_ses")
    if not os.path.isdir("./model_save_ses"):
        os.mkdir("./model_save_ses")
    try:
        saver = tf.train.Saver()
        saver.save(ses, './model_save_ses/model_save',
                   global_step=step, write_meta_graph=True)

        save_train_board_ses_pb(ses, step)
        prompt_green(
            "**model_ses {} saved, graph_info: {}".format(step, info))
        return True
    except Exception as ex:
        prompt_red("**model_ses {} failed to saved {}".format(step, ex))
        with ConsoleColor(ConsoleColor.RED):
            traceback.print_exc()
    return False
