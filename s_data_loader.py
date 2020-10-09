
import numpy as np
import os

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING", 
    "WALKING_UPSTAIRS", 
    "WALKING_DOWNSTAIRS", 
    "SITTING", 
    "STANDING", 
    "LAYING"
] 

loaded = False

# %% [markdown]
# ## Let's start by downloading the data: 

# %%
# Note: Linux bash commands start with a "!" inside those "ipython notebook" cells

def data_root():
    path = None
    f1 = "data"
    if os.path.isdir(f1):
        return os.path.abspath(f1) + "/"
    if path is None: 
        f2 = "../data"
        if os.path.isdir(f2):
            return os.path.abspath(f2) + "/"
    if path is None:
        print("Failed find data root") 
    return path     


# DATA_PATH = "data/"
DATA_PATH = data_root()


DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)

# %% [markdown]
# ## Preparing dataset:

# %%
TRAIN = "train/"
TEST = "test/"


# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()
    
    return np.transpose(np.array(X_signals), (1, 2, 0))


X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

# print("x_train: {}".format(X_train))
# print("x_test: {}".format(X_test))

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]], 
        dtype=np.int32
    )
    file.close()
    
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1


y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"


class DataHolder(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
    
        self.y_train = y_train
        self.y_test = y_test
        
        self.X_train_signals_paths = X_train_signals_paths
        self.X_test_signals_paths = X_test_signals_paths
    
        self.y_train_path = y_train_path
        self.y_test_path = y_test_path

        self.INPUT_SIGNAL_TYPES = INPUT_SIGNAL_TYPES
        self.LABELS = LABELS


def load_all():
    global loaded
    if not loaded:
        print("### Prepare loading all data...")
        for path in X_train_signals_paths:
            print("X_train_signals_path: {}".format(path))
        X_train = load_X(X_train_signals_paths)
        for path in X_test_signals_paths:
            print("X_test_signals_path: {}".format(path))
        X_test = load_X(X_test_signals_paths)

        y_train = load_y(y_train_path)
        print("y_train_path: {}".format(y_train_path))
        y_test = load_y(y_test_path)
        print("y_test_path: {}".format(y_test_path))

        print("### All data has been loaded.")
        loaded = True

    return DataHolder(X_train, X_test, y_train, y_test)
