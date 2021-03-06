
from math import sqrt # square root function
from math import acos
from re import search # inverse of cosinus function
import numpy as np
import os
import math

from s_defaults import default_sample_num, default_inputs

INPUT_SIGNAL_TYPES_6 = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

INPUT_SIGNAL_TYPES_7 = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_",
    "body_acc_mag_"
]

INPUT_SIGNAL_TYPES_8 = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_",
    "???_1",
    "???_2"
]

INPUT_SIGNAL_TYPES_9 = [
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
TRAIN = "train/"
TEST = "test/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)


# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []
    
    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        series = [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]
        if default_sample_num != 128:
            for i in range(0, len(series)):
                series[i] = series[i][:default_sample_num]
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in series]
        )
        file.close()
    # x_signals np array: shape(6, 7352, 128) default_sample_num
    # transpose to np as shape(7352, 128, 6) default_sample_num
    return np.transpose(np.array(X_signals), (1, 2, 0))


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
    
    # y_ np shape(7352, 1) , value is from 1 to 6
    # y_ -1 will decrease all value with -1,  so the value will become 0 to 5 (0-based), keep shape untouched
    # Substract 1 to each output class for friendly 0-based indexing 
    return y_ - 1


def find_inputs_num():
    di = os.environ['DATA_INPUTS_NUM']
    if di is not None:
        return int(di)
    return 9


def _load_inputs_9():
    print("load raw data or feature data for 9 inputs")
    X_train_signals_paths_9 = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES_9]
    X_test_signals_paths_9 = [
            DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES_9]
    X_train_signals_paths_9 = X_train_signals_paths_9
    X_test_signals_paths_9 = X_test_signals_paths_9

    for path in X_train_signals_paths_9:
        print("X_train_signals_path: {}".format(path))
    X_train = load_X(X_train_signals_paths_9)
    for path in X_test_signals_paths_9:
        print("X_test_signals_path: {}".format(path))
    X_test = load_X(X_test_signals_paths_9)
    return X_train, X_test


def _load_inputs_6():
    print("load raw data or feature data for 6 inputs")
    X_train_signals_paths_6 = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES_6]
    X_test_signals_paths_6 = [
            DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES_6]
    X_train_signals_paths_6 = X_train_signals_paths_6
    X_test_signals_paths_6 = X_test_signals_paths_6

    for path in X_train_signals_paths_6:
        print("X_train_signals_path: {}".format(path))
    X_train = load_X(X_train_signals_paths_6)
    for path in X_test_signals_paths_6:
        print("X_test_signals_path: {}".format(path))
    X_test = load_X(X_test_signals_paths_6)
    return X_train, X_test


def _magnitude_vector(vector3D):  # vector[X,Y,Z]
    return sqrt((vector3D**2).sum())  # eulidian norm of that vector

def _angle(vector1, vector2):
    vector1_mag = _magnitude_vector(vector1)  # euclidian norm of V1
    vector2_mag = _magnitude_vector(vector2)  # euclidian norm of V2

    # scalar product of vector 1 and Vector 2
    scalar_product = np.dot(vector1, vector2)
    # the cosinus value of the angle between V1 and V2
    cos_angle = scalar_product / float(vector1_mag * vector2_mag)

    # just in case some values were added automatically
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1

    angle_value = float(acos(cos_angle)) # the angle value in radian
    return angle_value  # in radian.

def _get_new_mag_body_feature(X_):
    # print("shape of _X", X_.shape)
    #shape of X_ (nc(1000+), ns(default_sample_num), val(default_inputs))
    nd3 = X_
    nc_len = len(nd3)
    ns_len = len(nd3[0])
    np_mag = np.zeros((nc_len, ns_len, 1))
    np_jerk = np.zeros((nc_len, ns_len, 1))
    np_angle = np.zeros((nc_len, ns_len, 1))
    for nc in range(0, nc_len):
        for ns in range(0, ns_len):
            nd3_cs = nd3[nc][ns]
            np_mag[nc][ns][0] = math.sqrt(
                nd3_cs[0] ** 2 + nd3_cs[1] ** 2 + nd3_cs[2] ** 2)

            if ns == 0:
                np_jerk[nc][ns][0] = np_mag[nc][ns][0]
            else:
                np_jerk[nc][ns][0] = np_mag[nc][ns][0] - np_mag[nc][ns-1][0]

            V2_Vector=np.array([nd3_cs[0], nd3_cs[1], nd3_cs[2]]) # mean values
            V1_Vector = np.array([0, 0, 1])
            np_angle[nc][ns][0] = _angle(V2_Vector, V1_Vector)

    return np_mag, np_jerk, np_angle


def _get_new_mag_total_feature(X_):
    # print("shape of _X", X_.shape)
    #shape of X_ (nc(1000+), ns(default_sample_num), val(default_inputs))
    nd3 = X_
    nc_len = len(nd3)
    ns_len = len(nd3[0])
    np_mag = np.zeros((nc_len, ns_len, 1))
    for nc in range(0, nc_len):
        for ns in range(0, ns_len):
            nd3_cs = nd3[nc][ns]
            np_mag[nc][ns][0] = math.sqrt(
                nd3_cs[3] ** 2 + nd3_cs[4] ** 2 + nd3_cs[5] ** 2)
    return np_mag


def _load_inputs_7():
    print("load raw data or feature data for 7 inputs")
    X_train, X_test = _load_inputs_6()

    print("prepare mag data on top of 3 existing data...")
    np_mag = _get_new_mag_body_feature(X_train)
    X_train = np.concatenate((X_train, np_mag), axis=2)

    np_mag = _get_new_mag_body_feature(X_test)
    X_test = np.concatenate((X_test, np_mag), axis=2)

    return X_train, X_test

def _load_inputs_8A():
    global INPUT_SIGNAL_TYPES_8
    INPUT_SIGNAL_TYPES_8 = [
        "body_acc_y_",
        "body_acc_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
        "body_acc_mag_",
        "body_acc_angle_"
    ]

    print("load raw data or feature data for 8A inputs: 6 orgin plus mag and angle")
    X_train, X_test = _load_inputs_6()

    print("prepare mag data on top of 3 existing data...")
    np_mag, _, np_angle = _get_new_mag_body_feature(X_train)
    X_train = np.concatenate((X_train, np_mag), axis=2)
    X_train = np.concatenate((X_train, np_angle), axis=2)

    np_mag, _, np_angle = _get_new_mag_body_feature(X_test)
    X_test = np.concatenate((X_test, np_mag), axis=2)
    X_test = np.concatenate((X_test, np_angle), axis=2)

    return X_train, X_test

def _load_inputs_8B():
    global INPUT_SIGNAL_TYPES_8
    INPUT_SIGNAL_TYPES_8 = [
        "body_acc_y_",
        "body_acc_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
        "body_acc_mag_",
        "body_acc_jerk_"
    ]
    print("load raw data or feature data for 8B inputs: 6 orgin plus mag and jerk")
    X_train, X_test = _load_inputs_6()

    print("prepare mag data on top of 3 existing data...")
    np_mag, np_jerk, _ = _get_new_mag_body_feature(X_train)
    X_train = np.concatenate((X_train, np_mag), axis=2)
    X_train = np.concatenate((X_train, np_jerk), axis=2)

    np_mag, np_jerk, _ = _get_new_mag_body_feature(X_test)
    X_test = np.concatenate((X_test, np_mag), axis=2)
    X_test = np.concatenate((X_test, np_jerk), axis=2)

    return X_train, X_test


def _load_inputs_8C():
    global INPUT_SIGNAL_TYPES_8
    INPUT_SIGNAL_TYPES_8 = [
        "body_acc_y_",
        "body_acc_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
        "body_acc_mag_",
        "total_acc_mag_"
    ]    
    print("load raw data or feature data for 8C inputs: 6 orgin plus two mag")
    X_train, X_test = _load_inputs_6()

    print("prepare mag data on top of 3 existing data...")
    np_mag = _get_new_mag_body_feature(X_train)
    X_train = np.concatenate((X_train, np_mag), axis=2)
    np_mag = _get_new_mag_total_feature(X_train)
    X_train = np.concatenate((X_train, np_mag), axis=2)

    print("prepare mag data on top of 3 existing data...")
    np_mag = _get_new_mag_body_feature(X_test)
    X_test = np.concatenate((X_test, np_mag), axis=2)
    np_mag = _get_new_mag_total_feature(X_test)
    X_test = np.concatenate((X_test, np_mag), axis=2)

    return X_train, X_test


def _load_lables_all():
    print("load label data...")
    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"

    y_train = load_y(y_train_path)
    print("y_train_path: {}".format(y_train_path))
    y_test = load_y(y_test_path)
    print("y_test_path: {}".format(y_test_path))
    return y_train, y_test


class DataHolder(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test
        self.LABELS = LABELS


def load_all():
    global loaded
    if not loaded:
        #load x info (raw sample data or feature data)
        inputs = find_inputs_num()
        print("### Prepare loading all data {} inputs...".format(inputs))
        if inputs == 6:
            X_train, X_test = _load_inputs_6()
        elif inputs == 7:
            X_train, X_test = _load_inputs_7()
        elif inputs == 8:
            X_train, X_test = _load_inputs_8A()
        else:
            X_train, X_test = _load_inputs_9()

        #load y info (label to which activities)
        y_train, y_test = _load_lables_all()

        print("### All data has been loaded.")
        loaded = True

    return DataHolder(X_train, X_test, y_train, y_test)


def get_input_names(inputs=default_inputs):
    if inputs == 6:
        return INPUT_SIGNAL_TYPES_6
    elif inputs == 7:
        return INPUT_SIGNAL_TYPES_7
    elif inputs == 8:
        return INPUT_SIGNAL_TYPES_8
    elif inputs == 9:
        return INPUT_SIGNAL_TYPES_9
    return []
