import tensorflow as tf
import time
import os
import numpy as np


def unique_path(log_dir):
    uniquename = time.strftime('log_%Y_%m_%d_%H_%M_%S')
    path = os.path.join(log_dir, uniquename)
    return path

def tensor_logs(abs_path,X_train):
    # abs_path = unique_path(log_dir)
    file_writter = tf.summary.create_file_writer(abs_path)
    
    with  file_writter.as_default():
        images = np.reshape(X_train,(-1,28,28,1))
        tf.summary.image("Hand writting digits sample",images,max_outputs=20,step=0)


