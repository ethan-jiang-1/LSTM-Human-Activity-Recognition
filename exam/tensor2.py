import tensorflow as tf
print("tensorflower version: {}".format(tf.VERSION))


def inspect(tobj, mark):
    print(mark, tobj)
    if hasattr(tobj, "name"):
        print(tobj.name, tobj.shape, tobj.dtype, tobj.op)
    else:
        print("object is not tf object")

# Tensor('Const:0' shape=(6, 2) dtype=int32)
my_tensor = tf.constant(0, shape=[6, 2])
inspect(my_tensor, "my_tensor")

my_dynamic_shape = tf.shape(my_tensor)
inspect(my_dynamic_shape, "my_dynamic_shape")
# -> Tensor('Shape:0' shape=(2,) dtype=int32)
# The shape of the tensor "Shape" is (2,) because my_tensor is a 2-D tensor
# so the dynamic shape is a 1-D tensor containing sizes of my_tensor dimensions
# and in this case, we have 2 dimensions.

my_reshaped_tensor = tf.reshape(my_tensor, [2, 3, 2])
inspect(my_reshaped_tensor, "my_reshaped_tensor")
# -> Tensor('Reshape:0' shape=(2, 3, 2) dtype=int32)

# To access a dynamic shape value, you need to run your graph and feed any placeholder that your tensor my depended upon:
my_eval = my_dynamic_shape.eval(session=tf.Session(), feed_dict={
    my_tensor: [[1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.]]
})

inspect(my_eval, "my_eval")
#

with tf.Session() as sess:
    my_eval1 = sess.run(my_dynamic_shape, feed_dict={
        my_tensor: [[1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.]]
    })
    inspect(my_eval1, "my_eval1")


pass
