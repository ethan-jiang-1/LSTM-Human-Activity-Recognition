# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor
from s_graph import inspect_graph

save_ses_enabled = False

class SessModelSaver(object):
    def __init__(self, sess, step, inputs=9):
        self.sess = sess
        self.step = step
        self.inputs = inputs
        self.model_dir = "model_save_{}_sess".format(self.inputs)

    def get_model_dir(self):
        return self.model_dir 

    def save_train_board_ses_pb(self):
        try:
            tf.train.write_graph(
                self.sess.graph_def, '', './{}/model_save-{}.pb'.format(self.model_dir, self.step), as_text=False)
        except Exception as ex:
            prompt_red("**model_ses {} failed to saved {}".format(self.step, ex))
            with ConsoleColor(ConsoleColor.RED):
                traceback.print_exc()

    def save(self):
        if not save_ses_enabled:
            return False

        info = inspect_graph("saved_model_ses")
        model_dir = "./" + self.model_dir
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        try:
            saver = tf.train.Saver()
            saver.save(self.sess, './{}/model_save'.format(self.model_dir),
                       global_step=self.step, write_meta_graph=True)

            self.save_train_board_ses_pb()
            prompt_green(
                "**model_ses {} saved, graph_info: {}".format(self.step, info))
            return True
        except Exception as ex:
            prompt_red("**model_ses {} failed to saved {}".format(self.step, ex))
            with ConsoleColor(ConsoleColor.RED):
                traceback.print_exc()
        return False
