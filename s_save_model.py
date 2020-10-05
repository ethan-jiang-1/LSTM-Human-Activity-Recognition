# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor
from x_simple_save import simple_save


# tf.enable_resource_variables()
graph = tf.get_default_graph()


def inspect_graph(mark):
    if mark is not None:
        prompt_yellow(mark)
    for op in graph.get_operations():
        if op.name.find("my_") == 0:
            prompt_blue(op.name, op.type, op.values())
    return "len({})".format(len(graph.get_operations()))


et_dir_name = '/tmp/LSTM_logs'
if os.path.isdir(et_dir_name):
    shutil.rmtree(et_dir_name)
os.mkdir(et_dir_name)
et_op_merge_all = tf.summary.merge_all()
et_summary_writer = None

def _get_summary_write(ses):
    global et_summary_writer
    if et_summary_writer is None:
        et_summary_writer = tf.summary.FileWriter(et_dir_name, ses.graph)
    return et_summary_writer

def _prompt_board(ses, step):
    if step % 100 != 0:
        return
    prompt_yellow("Run the command line: --> tensorboard --logdir={} "
                  "\nThen open http://localhost:6006/ into your web browser".format(et_dir_name))


def export_tensorboard(ses, step, x, y, nx, ny):
    writer = _get_summary_write(ses)
    try:
        # summary  = et_op_merge_all.eval(session=ses, feed_dict={})
        summary = ses.run(et_op_merge_all, feed_dict={x: nx, y: ny})
        if summary is not None:
            writer.add_summary(summary, step)
            _prompt_board(ses, step)
        return True
    except Exception as e:
        prompt_red("\n\n** export_tensorboard: {}".format(e))
        with ConsoleColor(ConsoleColor.RED):
            traceback.print_exc()
    return False

def _add_name_to_tensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)


# aux functions


def _prepare_save_pb(ses, step, name, nx, ny):
    dir_name = "./" + name
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    else:
        os.mkdir(dir_name)
    tx = nx
    # if nx is not None:
    #     if not hasattr(nx, "op"):
    #         tx = _add_name_to_tensor(nx, "my_x_input_step")
    ty = ny
    # if ny is not None:
    #     if not hasattr(nx, "op"):
    #         ty = _add_name_to_tensor(ny, "my_y_output_step")

    return dir_name, tx, ty


def save_model_pb(ses, step, name, nx=None, ny=None):
    dir_name, tx, ty = _prepare_save_pb(ses, step, name, nx, ny)
    prompt_green("tx: {}".format(tx))
    prompt_green("ty: {}".format(ty))

    try:
        inspect_graph("saved_model")
        # tf.saved_model.simple_save(ses,
        #                             dir_name,
        #                             inputs={"my_inputs": tx},
        #                             outputs={"my_outputs": ty})
        simple_save(ses,
                    dir_name,
                    inputs={"my_inputs": tx},
                    outputs={"my_outputs": ty})
        prompt_green("\n\n**model {} saved.".format(step))
    except Exception as ex:
        prompt_red("\n\n**model {} failed to saved: {}".format(step, ex))
        with ConsoleColor(ConsoleColor.RED):
            traceback.print_exc()
    return False


def save_model_ses(ses, step):
    if not os.path.isdir("./model_save_ses"):
        os.mkdir("./model_save_ses")
    try:
        info = inspect_graph("saved_model_ses")
        saver = tf.train.Saver()
        saver.save(ses, './model_save_ses/model_save',
                   global_step=step, write_meta_graph=True)
        tf.train.write_graph(
            ses.graph_def, '', './model_save_ses/model_save-{}.pb'.format(step), as_text=False)
        prompt_green(
            "**model_ses {} saved, graph_info: {}".format(step, info))
        return True
    except Exception as ex:
        prompt_red("**model_ses {} failed to saved {}".format(step, ex))
        with ConsoleColor(ConsoleColor.RED):
            traceback.print_exc()
    return False
