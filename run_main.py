
import os
import shutil
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

print("import tf")
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)

from s_save_model import save_model_pb, save_model_ses
from s_graph import inspect_graph, get_summary_writer, add_summary
from s_console_prompt import prompt_yellow, prompt_blue, prompt_green, prompt_red, prompt_progress
from s_data_loader import load_all
# load dataset from data_loader
dh = load_all()
X_train = dh.X_train
X_test = dh.X_test
y_train = dh.y_train
y_test = dh.y_test
LABELS = dh.LABELS

list_gpu = tf.config.experimental.list_physical_devices('GPU')
prompt_yellow("Num GPUs Available: ", list_gpu, len(list_gpu))
tf.enable_resource_variables()
tf.logging.set_verbosity(tf.logging.ERROR)

# %% [markdown]
# ## Additionnal Parameters:
# 
# Here are some core parameter definitions for the training. 
# 
# For example, the whole neural network's structure could be summarised by enumerating those parameters and the fact that two LSTM are used one on top of another (stacked) output-to-input as hidden layers through time steps. 

# %%
# Input Data 


training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep


# LSTM Neural Network's internal structure
n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)


# Training 

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 30000  # To show test set accuracy during training


# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("training (X shape, y shape, every X's mean, every X's standard deviation)")
print(X_train.shape, y_train.shape, np.mean(X_train), np.std(X_train))
print("test (X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape, dtype=np.float32)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes 
    # e.g.: 
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes, dtype=np.int32)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


# %% [markdown]
cnc = tf.constant(n_classes, name="my_n_classes")
cns = tf.constant(n_steps, name="my_n_steps")
cni = tf.constant(n_input, name="my_n_input")
cnh = tf.constant(n_hidden, name="my_n_hidden")

cbatch_xs = extract_batch_size(X_test, 1, 1)
cbatch_ys = extract_batch_size(y_test, 1, 1)
cbatch_ys_oh = one_hot(cbatch_ys)

ctx = tf.constant(cbatch_xs, name="my_c_input")
cty = tf.constant(cbatch_ys_oh, name="my_c_output")

# ## Utility functions for training:    
# %%
def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset, some of the credits goes to 
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    #_X = tf.reshape(_X, [-1, n_input]) 
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier, 
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


# %% [markdown]
# ## Let's get serious and build the neural network:
inspect_graph("start")

# Graph input/output
prompt_progress("Input/Output")
with tf.name_scope("Input"):
    # 128 steps 9 input
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name="my_x_input")
with tf.name_scope("Output"):
    # 6 classified result
    y = tf.placeholder(tf.float32, [None, n_classes], name="my_y_output")

prompt_progress("Model")
with tf.name_scope("Model"):
    # Graph weights
    weights = {
        # Hidden layer weights
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden], name="weights_hidden")),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0), name="weights_out")
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden]), name="biases_hidden"),
        'out': tf.Variable(tf.random_normal([n_classes]), name="biases_out")
    }    
    pred = LSTM_RNN(x, weights, biases)
    pred = tf.identity(pred, name="my_pred")

prompt_progress("Loss")
with tf.name_scope("Loss"):
    # Loss, optimizer and evaluation
    _l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()) # L2 loss prevents this overkill neural network to overfit the data

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + _l2 # Softmax loss

prompt_progress("Optimizer")
with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
    # optimizer = tf.identity(optimizer, name="my_optimizer")

prompt_progress("Accuray")
with tf.name_scope("Accuray"):
    _correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1), name="_correct_pred")
    accuracy = tf.reduce_mean(tf.cast(_correct_pred, tf.float32), name="accuracy")
    accuracy = tf.identity(accuracy, name="my_accuracy")

# %% [markdown]
prompt_progress("init")
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
# ## Hooray, now train the neural network:

# %%
# To keep track of training's performance
# Launch the graph
prompt_progress("SessionInit")
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True, device_count={'GPU': len(list_gpu)}))
sess.run(init)
inspect_graph("after_init")

test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

writer = get_summary_writer(sess)


def save_model_pred(sess, step):
    prompt_yellow("save_model_pred {}".format(step))
    po_batch_one_xs = extract_batch_size(X_test, step, 1)
    po_batch_one_ys = extract_batch_size(y_test, step, 1)
    po_batch_one_ys_oh = one_hot(po_batch_one_ys)
    po_one_hot_predictions, po_accuracy, po_final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: po_batch_one_xs,
            y: po_batch_one_ys_oh
        }
    )
    prompt_yellow("pred", po_one_hot_predictions, po_one_hot_predictions.argmax(1))
    prompt_yellow("actual", po_batch_one_ys_oh, po_batch_one_ys_oh.argmax(1))
    prompt_yellow("accuracy", po_accuracy)
    prompt_yellow("loss", po_final_loss)    
    save_model_pb(sess, step, "model_save_" + str(step), x, y, po_batch_one_xs, po_batch_one_ys_oh, ctx, cty)


# Perform Training steps with "batch_size" amount of example data at each loop
prompt_progress("SessionLoopStart")
step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(X_train, step, batch_size)
    batch_ys = extract_batch_size(y_train, step, batch_size)
    batch_ys_oh = one_hot(batch_ys)

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys_oh
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Training iter #" + str(step*batch_size) + ":   Batch Loss = " + "{:.6f}".format(loss) + ", Accuracy = {}".format(acc))
        if step % 400 == 100:
            save_model_pred(sess, step)
        if step % 100 == 0:
            save_model_ses(sess, step)
            add_summary(sess, step, merged_summary_op, feed_dict={
                x: batch_xs,
                y: batch_ys_oh
            })

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + "Batch Loss = {} , Accuracy = {} @Step:{}".format(loss, acc, step))

    step += 1


print("Optimization Finished!")
step += 1
save_model_pred(sess, step)


# Final prediction: Accuracy for all test data
y_test_oh = one_hot(y_test)
one_hot_predictions, final_accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: y_test_oh
    }
)
test_losses.append(final_loss)
test_accuracies.append(final_accuracy)
print("FINAL RESULT: " + "Batch Loss = {}".format(final_loss) + ", Accuracy = {}".format(final_accuracy))


# %% [markdown]
# ## Training is good, but having visual insight is even better:
# 
# Okay, let's plot this simply in the notebook for now.

# %%
# (Inline plots: )
# get_ipython().run_line_magic('matplotlib', 'inline')

font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 12
}
matplotlib.rc('font', **font)

width = 9
height = 6
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()

# %% [markdown]
# ## And finally, the multi-class confusion matrix and metrics!

# %%
# Results

predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*final_accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to {} of total test data):".format(len(X_test)))
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 9
height = 6
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


sess.close()
