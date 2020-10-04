import tensorflow as tf
print("tensorflower version: {}".format(tf.VERSION))


def inspect(tobj, mark):
    print(mark, tobj)
    if hasattr(tobj, "name") and hasattr(tobj, "shape"):
        print(tobj.name, tobj.shape, tobj.dtype, tobj.op)
    else:
        print("object is not tf object")


seq_max_length = 100
rnn_inputs = []

my_placeholder = tf.placeholder(tf.float32, shape=[None, seq_max_length])

cell = tf.nn.rnn_cell.LSTMCell(num_units=50)
inspect(cell, "cell")

# The batch_size is dynamic
#init_state = cell.zero_state(batch_size=???, tf.float32)
init_state = cell.zero_state(batch_size=10, dtype=tf.float32)
outputs, encoder_final_state = tf.nn.dynamic_rnn(
    cell, rnn_inputs, initial_state=init_state)

pass
