# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor
from s_graph import inspect_graph

save_pred_enabled = True

class PredModelSaver(object):
    def __init__(self, sess, step, pred, x, inputs=9):
        self.sess = sess
        self.step = step
        self.pred = pred
        self.x = x
        self.inputs = inputs
        self.model_dir = "model_save_{}_{}".format(self.inputs, self.step)
    
    def get_model_dir(self):
        return self.model_dir

    def save(self):
        if not save_pred_enabled:
            return False

        self._save_pred_model_pb()
        return True

    def _prepare_save_dir(self):
        dir_name = "./" + self.model_dir
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        else:
            os.mkdir(dir_name)
        return dir_name

    def _save_pred_model_pb(self):
        prompt_yellow("_save_pred_model_pb {}".format(self.step))
        inspect_graph("saved_model_0")
        dir_name = self._prepare_save_dir()
        try:
            from xt_tf.xp_simple_save import simple_save_ex
            simple_save_ex(self.sess,
                        dir_name,
                        inputs={"x": self.x},
                        outputs={"y": self.pred})
            prompt_green("\n\n**model {} saved.".format(self.step))
            inspect_graph("saved_model_1")
            return True
        except Exception as ex:
            prompt_red("\n\n**model {} failed to saved: {}".format(self.step, ex))
            with ConsoleColor(ConsoleColor.RED):
                traceback.print_exc()
        return False
