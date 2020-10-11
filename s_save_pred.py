# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor
from s_graph import inspect_graph
from sklearn import metrics
import numpy as np

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


class PredResultSaver(object):
    def __init__(self, sess, step, acc, loss, pred_test=None, y_test=None, inputs=9):
        self.sess = sess
        self.step = step
        self.acc = acc
        self.loss = loss
        self.pred_test = pred_test
        self.y_test = y_test
        self.inputs = inputs
        self.model_dir = "model_save_{}_{}".format(self.inputs, self.step)

    def save(self):
        if not os.path.isdir(self.model_dir):
            return False

        try:
            with open(self.model_dir + "/pred_summary.txt", "w+") as f:
                f.write("Batch Loss = {:4f} , Accuracy = {:.4f} @Step:{}\n".format(self.loss, self.acc, self.step))

                if self.pred_test is not None and self.y_test is not None:
                    y_test = self.y_test
                    pred_test = self.pred_test

                    f.write("\n")
                    f.write("Precision: {:.4f}%\n".format(100*metrics.precision_score(y_test, pred_test, average="weighted")))
                    f.write("Recall: {:4f}%\n".format(100*metrics.recall_score(y_test, pred_test, average="weighted")))
                    f.write("f1_score: {:.4f}%\n".format(100*metrics.f1_score(y_test, pred_test, average="weighted")))

                    f.write("\n")
                    f.write("Confusion Matrix:\n")
                    confusion_matrix = metrics.confusion_matrix(y_test, pred_test)
                    f.write(str(confusion_matrix))
                    f.write("\n")
                    
                    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
                    f.write("\n")
                    f.write(str(normalised_confusion_matrix))
                    f.write("\n")
                    f.write("Note: training and testing data is not equally distributed amongst classes, \n")
                    f.write("so it is normal that more than a 6th of the data is correctly classifier in the last category.\n")

                return True
        except Exception as ex:
            prompt_red("\n\n**model pred {} failed to saved: {}".format(self.step, ex))
            with ConsoleColor(ConsoleColor.RED):
                traceback.print_exc()
        return False            

