from src.utils.common import get_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,save_model
import os
import tensorflow as tf
from src.utils.tensor_logs import unique_path,tensor_logs
import argparse



def training(config_path):

    config = get_config(config_path)
    val_thresh = config['params']['validation_datasize']
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(val_thresh)

    tensor_dir = config['logs']['tensorlog_dir']
    tensor_path = unique_path(tensor_dir)
    tensor_logs(tensor_path,X_train[20:30])

    LOSS_FUNCTION = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    NUM_CLASSES = config['params']['num_classes']

    model_clf = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    val_set = (X_valid, y_valid)
    EPOCH = config['params']['epochs']

    tensorboard_cb = tf.keras.callbacks.TensorBoard(tensor_path)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    os.makedirs('checkpoint/models',exist_ok = True)
    CKPT_path="checkpoint/models/weights-improvement-{epoch:02d}.h5"
    # CKPT_path = "checkpoint/models/model_ckpt.h5"
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path,epoch = 5)
    CALLBACK = [tensorboard_cb,early_stop_cb,model_checkpoint_cb]

    if len(os.listdir('checkpoint/models')) == 0:
        history = model_clf.fit(X_train, y_train,epochs = EPOCH, callbacks=CALLBACK, validation_data = val_set)
        artifact_dir = config['artifacts']['artifacts_dir']
        model_dir  = config['artifacts']['model_dir']
        path = os.path.join(artifact_dir,model_dir)
        os.makedirs(path,exist_ok=True)
        model_name = config['artifacts']['model_name']
    
        save_model(model_clf,model_name,path)
    else:
        ckpt_dir_path = 'checkpoint/models'
        ckpt_model_path = os.listdir(ckpt_dir_path)[-1]
        ckpt_full_path = os.path.join(ckpt_dir_path,ckpt_model_path)
        ckpt_model = tf.keras.models.load_model(ckpt_full_path)
        # for file in CKPT_path[:-1]

        history = ckpt_model.fit(X_train, y_train,epochs = EPOCH, callbacks=CALLBACK, validation_data = val_set)

        artifact_dir = config['artifacts']['artifacts_dir']
        model_dir  = config['artifacts']['model_dir']
        path = os.path.join(artifact_dir,model_dir)
        os.makedirs(path,exist_ok=True)
        model_name = config['artifacts']['model_name']
        save_model(ckpt_model,model_name,path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)
    # tf.tensorboard --'config'='./logs' --'port'=6006
   
