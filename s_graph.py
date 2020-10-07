# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor


# tf.enable_resource_variables()
graph = tf.get_default_graph()


def inspect_graph(mark, sess=None):
    if mark is not None:
        prompt_yellow(mark)

    cg = graph
    if sess is not None:
        cg = sess.graph
    for op in cg.get_operations():
        if op.name.find("my_") != -1:
            prompt_blue(op.name, op.type, op.values())
    return "len({})".format(len(cg.get_operations()))


et_dir_name = '/tmp/LSTM_logs'
if os.path.isdir(et_dir_name):
    shutil.rmtree(et_dir_name)
os.mkdir(et_dir_name)
et_summary_writer = None


def get_summary_writer(sess):
    global et_summary_writer
    if et_summary_writer is None:
        et_summary_writer = tf.summary.FileWriter(et_dir_name, sess.graph)
    return et_summary_writer


def _prompt_board(sess, step):
    if step % 100 != 0:
        return
    prompt_yellow("Run the command line: --> tensorboard --logdir={} "
                  "\nThen open http://localhost:6006/ into your web browser".format(et_dir_name))


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
