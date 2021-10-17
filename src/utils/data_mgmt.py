import tensorflow as tf


def get_data(val_thresh):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    # split train and validation

    X_valid,X_train = X_train_full[:val_thresh]/255. , X_train_full[val_thresh:]/255.
    y_valid,y_train = y_train_full[:val_thresh],y_train_full[val_thresh:]
    X_test = X_test/255.

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    