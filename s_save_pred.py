# Version 1.0.0 (some previous versions are used in past commits)
import tensorflow as tf
import shutil
import os

import traceback
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red
from s_console_prompt import ConsoleColor
from s_graph import inspect_graph

save_pred_enabled = True


# def _add_name_to_tensor(someTensor, theName):
#     return tf.identity(someTensor, name=theName)


def _prepare_save_dir(ses, step, name):
    dir_name = "./" + name
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    else:
        os.mkdir(dir_name)
    return dir_name


def _prepare_io_tensor(ses, x, y, vx, vy):
    #with tf.name_scope("Input"):
    #    tx = _add_name_to_tensor(vx, "my_x_input")
    #with tf.name_scope("Output"):
    #    ty = _add_name_to_tensor(vy, "my_y_output")
    # named_input_output = True
    tsx = tf.convert_to_tensor(vx)
    tsy = tf.convert_to_tensor(vy)
    cvx, cvy = ses.run([tsx, tsy], feed_dict = {x:vx, y:vy}) 

    tx = ses.graph.get_tensor_by_name("my_input:0")
    if tx is None:
        tx = tf.identity(tsx, name="my_input")
    
    ty = ses.graph.get_tensor_by_name("my_output:0")
    if ty is None:
        ty = tf.identity(tsy, name="my_output")

    prompt_yellow("_prepare_io_tensor, value(cvx,cvy) {} {}".format(cvx, cvy)) 
    prompt_yellow("_prepare_io_tensor, finial(tx, ty) {} {}".format(tx, ty))

    return tx, ty


def _save_pred_model_pb(ses, step, name, x, y, vx, vy, cx, cy):
    prompt_yellow("_save_pred_model_pb {}".format(step))
    inspect_graph("saved_model_0")
    dir_name = _prepare_save_dir(ses, step, name)
    # tx, ty = _prepare_io_tensor(ses, x, y, vx, vy)
    # prompt_green("tx: {}".format(tx))
    # prompt_green("ty: {}".format(ty))
    try:
        # attemp1 
        # tf.saved_model.simple_save(ses,
        #                              dir_name,
        #                              inputs={"my_inputs": tx},
        #                              outputs={"my_outputs": ty})
        
        # attemp2 -- working (but...)
        # from xt_tf.xp_simple_save import simple_save
        # simple_save(ses,
        #             dir_name,
        #             inputs={"x": tx},
        #             outputs={"y": ty})
        
        # attemp3
        # builder = tf.saved_model.builder.SavedModelBuilder(dir_name)
        # builder.add_meta_graph_and_variables(
        #     ses, [tf.saved_model.tag_constants.SERVING])
        # builder.save()

        # attemp4 - working (but...)
        # from xt_tf.xp_simple_save import simple_save_ex
        # simple_save_ex(ses,
        #          dir_name,
        #          inputs={"x": cx},
        #          outputs={"y": cy})

        # attemp5 (working?!)
        from xt_tf.xp_simple_save import simple_save_ex
        simple_save_ex(ses,
                 dir_name,
                 inputs={"x": x},
                 outputs={"y": y})

        # attemp6
        # ti_input_x = tf.saved_model.utils.build_tensor_info(x)
        # ti_output_y = tf.saved_model.utils.build_tensor_info(y)
        # from xt_tf.xp_simple_save import simple_save_ex
        # simple_save_ex(ses,
        #          dir_name,
        #          inputs={"x": ti_input_x},
        #          outputs={"y": ti_output_y})

        prompt_green("\n\n**model {} saved.".format(step))
        inspect_graph("saved_model_1")
        return True
    except Exception as ex:
        prompt_red("\n\n**model {} failed to saved: {}".format(step, ex))
        with ConsoleColor(ConsoleColor.RED):
            traceback.print_exc()
    return False


class PredModelSaver(object):
    def __init__(self, sess, step, pred, one_xs, one_ys_oh, x, y, ctx, cty):
        self.sess = sess
        self.step = step
        self.pred = pred

        self.one_xs = one_xs
        self.one_ys_oh = one_ys_oh

        self.x = x
        self.y = y
        self.ctx = ctx
        self.cty = cty

    def save(self):
        if not save_pred_enabled:
            return False
        
        sess = self.sess
        step = self.step
        pred = self.pred 

        one_xs = self.one_xs
        one_ys_oh = self.one_ys_oh

        x = self.x
        y = self.y 
        ctx = self.ctx
        cty = self.cty

        prompt_yellow("save_model_pred {}".format(step))
        pred_mv = sess.run(
            [pred],
            feed_dict={
                x: one_xs,
                y: one_ys_oh
            }
        )
        prompt_yellow("pred  ", pred_mv)
        prompt_yellow("actual", one_ys_oh)
        _save_pred_model_pb(sess, step, "model_save_" + str(step), x, y, one_xs, one_ys_oh, ctx, cty)
        return True
