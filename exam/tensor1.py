import tensorflow as tf
print("tensorflower version: {}".format(tf.VERSION))

def inspect(tobj, mark):
    print(mark, tobj)
    print(tobj.name, tobj.shape, tobj.dtype, tobj.op)
    

# TensorFlow's way
p = tf.placeholder(tf.float32, shape=[], name="p")
inspect(p, "placeholder")

v2 = tf.Variable(2. , name="v2")
inspect(v2, "variable")

add1 = tf.add(p, v2)
inspect(add1, "add1")

add2 = tf.add(2, 2)
inspect(add2, "add2")  # => Tensor("Add:0", shape=(), dtype=int32)

my_tensor = tf.constant(0., shape=[6,3,7])
inspect(my_tensor, "const") # -> Tensor("Const:0", shape=(6, 3, 7), dtype=float32)

my_placeholder = tf.placeholder(tf.float32, shape=[None, 2])
inspect(my_placeholder, "placeholder1")
# -> [?, 2]

my_placeholder.set_shape([8, 2])
inspect(my_placeholder, "placeholder1")


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # From the moment we initiliaze variables, until the end of the Session's scope
  # We can access variables
  print(sess.run(v2)) # -> 2.

  # On the other hand, intermediate tensors has to be recalculated 
  # each time you want to access its value
  print(sess.run(a, feed_dict={p: 3})) # -> 5.

  # Even if calculated once, the value of a is no more accessible
  # the value of a has been freed off the memory
  try:
    sess.run(a) # Error: "You must feed a value for placeholder tensor 'p' with dtype float"
  except Exception as e:
    print(e)

pass
