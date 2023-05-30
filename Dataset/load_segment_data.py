import os
import numpy as np
import logging
from Dataset import data_loader
from Dataset.classifier_tools import prepare_inputs_deep_learning
logger = logging.getLogger(__name__)


def load(config):
    Data = {}
    window_len = 40
    stride = 20
    val_size = 2
    problem = config['data_dir'].split('/')[-1]

    if os.path.exists(config['data_dir'] + '/' + problem + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + problem + '.npy', allow_pickle=True)

        Data['max_len'] = Data_npy.item().get('max_len')
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['val_data'] = Data_npy.item().get('val_data')
        Data['val_label'] = Data_npy.item().get('val_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))
    else:
        if problem == "ActivityRecognition":
            all_file = config['data_dir'] + '/' + problem + ".txt"
            train_data, test_data = data_loader.load_activity(all_file, "Clean")

        elif problem == "FordChallenge":
            val_size = 10
            train_file = config['data_dir'] + '/' + problem + "_TRAIN.csv"
            test_file = config['data_dir'] + '/' + problem + "_TEST.csv"
            train_data = data_loader.load_ford_data(train_file, "Clean")
            test_data = data_loader.load_ford_data(test_file, "Clean")

        # Emotiv Datasets
        else:
            train_file = config['data_dir'] + '/' + problem + "_TRAIN.csv"
            test_file = config['data_dir'] + '/' + problem + "_TEST.csv"
            train_data = data_loader.load_segmentation_data(train_file, "Clean")
            test_data = data_loader.load_segmentation_data(test_file, "Clean")

        X_train, y_train, X_val, y_val, X_test, y_test = prepare_inputs_deep_learning(train_inputs=train_data,
                                                                                      test_inputs=test_data,
                                                                                      window_len=window_len,
                                                                                      stride=stride, val_size=val_size)

        logger.info("{} samples will be used for training".format(len(y_train)))
        logger.info("{} samples will be used for validation".format(len(y_val)))
        logger.info("{} samples will be used for testing".format(len(y_test)))
        # Reshape to = (sample, dim, len)
        Data['max_len'] = window_len
        Data['train_data'] = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
        Data['train_label'] = y_train

        Data['val_data'] = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[1])
        Data['val_label'] = y_val

        Data['test_data'] = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])
        Data['test_label'] = y_test

        np.save(config['data_dir'] + "/" + problem, Data, allow_pickle=True)

    return Data