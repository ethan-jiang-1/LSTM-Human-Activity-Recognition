import numpy as np

def inspect_data(X_train, X_test, y_train, y_test):
    print("\n**")
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("training (X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_train.shape, y_train.shape, np.mean(X_train), np.std(X_train))
    print("test (X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    # 7352 training series (with 50% overlap between each serie)
    training_data_count = len(X_train)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep

    print("training_data_count: ", training_data_count)
    print("test_data_count", test_data_count)
    print("n_steps", n_steps)
    print("n_input", n_input)
    print("**\n")
