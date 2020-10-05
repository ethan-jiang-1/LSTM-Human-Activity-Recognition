# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor
from s_graph import inspect_graph
from x_simple_save import simple_save


def _add_name_to_tensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)


# aux functions


def _prepare_save_dir(ses, step, name):
    dir_name = "./" + name
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    else:
        os.mkdir(dir_name)
    return dir_name


def _prepare_save_tensor(nx, ny):
    tx = nx
    if nx is not None:
        if not hasattr(nx, "op"):
            tx = _add_name_to_tensor(nx, "my_x_input_step")
    ty = ny
    if ny is not None:
        if not hasattr(nx, "op"):
            ty = _add_name_to_tensor(ny, "my_y_output_step")

    return tx, ty


def save_model_pb(ses, step, name, nx=None, ny=None):
    dir_name = _prepare_save_dir(ses, step, name)
    tx, ty = _prepare_save_tensor(nx, ny)
    # prompt_green("tx: {}".format(tx))
    # prompt_green("ty: {}".format(ty))

    try:
        inspect_graph("saved_model")

        # attemp1 
        # tf.saved_model.simple_save(ses,
        #                             dir_name,
        #                             inputs={"my_inputs": tx},
        #                             outputs={"my_outputs": ty})
        
        # attemp2
        simple_save(ses,
                     dir_name,
                     inputs={"my_inputs": tx},
                     outputs={"my_outputs": ty})
        
        # attemp3
        # builder = tf.saved_model.builder.SavedModelBuilder(dir_name)
        # builder.add_meta_graph_and_variables(
        #     ses, [tf.saved_model.tag_constants.SERVING])
        # builder.save()

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
