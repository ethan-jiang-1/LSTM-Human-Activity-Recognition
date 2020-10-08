# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor


# tf.enable_resource_variables()
default_graph = tf.get_default_graph()


def inspect_graph(mark, sess=None, graph=None):
    if mark is not None:
        prompt_yellow("inspect_graph:" + mark)

    cg = default_graph
    if sess is not None:
        cg = sess.graph
    if graph is not None:
        cg = graph

    if hasattr(cg, "get_operations"):
        prompt_blue("total ops: {}".format(len(cg.get_operations())))
        for op in cg.get_operations():
            if op.name.find("my_") != -1:
                prompt_blue(op.name, op.type, op.values())
        return "len({})".format(len(cg.get_operations()))
    else:
        prompt_red("not a graph")
        return "???"


et_dir_name = None
et_summary_writer = None

def get_log_folder():
    global et_dir_name
    if et_dir_name is not None:
        return et_dir_name

    try:
        dir_name = '/tmp/LSTM_logs'
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        et_dir_name = dir_name
    except Exception as ex:
        print(ex)
        dir_name = os.path.abspath("/tmp")
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        et_dir_name = dir_name
    return et_dir_name


def get_summary_writer(sess):
    global et_summary_writer
    if et_summary_writer is None:
        dir_name = get_log_folder()
        et_summary_writer = tf.summary.FileWriter(dir_name, sess.graph)
    return et_summary_writer


def _prompt_board(sess, step):
    if step % 100 != 0:
        return
    prompt_yellow("Run the command line: --> tensorboard --logdir={} "
                  "\nThen open http://localhost:6006/ into your web browser".format(get_log_folder()))


def add_summary(sess, step, merged_summary_op, feed_dict=None):
    writer = get_summary_writer(sess)
    try:
        summary = sess.run(merged_summary_op, feed_dict=feed_dict)
        if summary is not None:
            writer.add_summary(summary, step)
            _prompt_board(sess, step)
        return True
    except Exception as e:
        prompt_red("\n\n** export_tensorboard: {}".format(e))
        with ConsoleColor(ConsoleColor.RED):
            traceback.print_exc()
    return False
