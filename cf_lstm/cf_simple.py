
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np

import s_data_loader as data_loader
dt = data_loader.load_all()


X_train = dt.X_train
X_test = dt.X_test
y_train = dt.y_train
y_test = dt.y_test


training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep
n_classes = 6

print(training_data_count, test_data_count, n_steps, n_input)


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
    batch_s = np.empty(shape)

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
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


step = 1
batch_size = 1500
batch_xs = extract_batch_size(X_train, step, batch_size)
batch_ys = extract_batch_size(y_train, step, batch_size)
batch_ys_oh = one_hot(batch_ys)
print(batch_xs.shape)
print(batch_ys.shape)
print(batch_ys_oh.shape)
