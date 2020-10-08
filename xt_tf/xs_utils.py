# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from s_console_prompt import prompt_yellow, prompt_blue  # , prompt_green, prompt_red


# tf.enable_resource_variables()
graph = tf.get_default_graph()


def inspect_graph(mark, sess=None):
    if mark is not None:
        prompt_yellow("inspect_graph:" + mark)

    cg = graph
    if sess is not None:
        cg = sess.graph
    for op in cg.get_operations():
        if op.name.find("my_") != -1:
            prompt_blue(op.name, op.type, op.values())
    return "len({})".format(len(cg.get_operations()))
