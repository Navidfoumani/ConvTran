"""
Written by 
"""
import logging

import numpy as np
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pickle
import pandas as pd
import time

# Project modules
from configuration import Configuration, setup, Initialization
from Datasets.UEA.data import load_ts, split_dataset, dataset_class, long_dataset_class, collate, process_ts_data
from transformers import model_factory
from optimizers import get_optimizer
from Model.loss import get_loss_module
from running import SupervisedRunner, validate
from utils import utils
from Datasets.Segmentation import data_loader
from Datasets.Segmentation.classifier_tools import prepare_inputs_deep_learning

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")

def load_UEA_data(config):
    # Build data
    Data = {}
    problem = config['data_dir'][13:]
    if os.path.exists(config['data_dir'] + '/' + problem + '.pickle'):
        logger.info("Loading preprocessed data ...")
        with open(config['data_dir'] + '/' + problem + '.pickle', 'rb') as handle:
            Data = pickle.load(handle)

        train_label = Data['train_label']
        val_label = Data['val_label']
        test_label = Data['test_label']

        logger.info("{} samples will be used for training".format(len(train_label)))
        logger.info("{} samples will be used for validation".format(len(val_label)))
        logger.info("{} samples will be used for testing".format(len(test_label)))

    else:
        logger.info("Loading and preprocessing data ...")
        train_data, train_label, test_data, test_label, max_seq_len = load_ts(config['data_dir'])
        # normalizer = Normalizer('standardization')
        # train_data = normalizer.normalize(train_data)
        # test_data = normalizer.normalize(test_data)
        Data['max_len'] = max_seq_len
        Data['All_train_data'] = train_data
        Data['All_train_label'] = train_label
        train_data, train_label, val_data, val_label = split_dataset(train_data, train_label, config['val_ratio'])

        logger.info("{} samples will be used for training".format(len(train_label)))
        logger.info("{} samples will be used for validation".format(len(val_label)))
        logger.info("{} samples will be used for testing".format(len(test_label)))

        Data['train_data'] = train_data
        Data['train_label'] = train_label

        Data['val_data'] = val_data
        Data['val_label'] = val_label

        Data['test_data'] = test_data
        Data['test_label'] = test_label

        with open(config['data_dir'] + '/' + problem + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return Data


def load_segment_data(config):
    Data = {}
    window_len = 40
    stride = 20
    val_size = 2
    problem = config['data_dir'][22:]

    if os.path.exists(config['data_dir'] + '/' + problem + '.pickle'):
        with open(config['data_dir'] + '/' + problem + '.pickle', 'rb') as handle:
            Data = pickle.load(handle)

        '''
        Data['train_data'] = process_ts_data(Data['train_data'], Data['max_len'], normalise=False)
        Data['val_data'] = process_ts_data(Data['val_data'], Data['max_len'], normalise=False)
        Data['test_data'] = process_ts_data(Data['test_data'], Data['max_len'], normalise=False)

        Data['train_label'] = np.squeeze(Data['train_label'].to_numpy())
        Data['val_label'] = np.squeeze(Data['val_label'].to_numpy())
        Data['test_label'] = np.squeeze(Data['test_label'].to_numpy())
        '''

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    else:
        if (problem == "ActivityRecognition"):
            all_file = config['data_dir'] + '/' + problem + ".txt"
            train_data, test_data = data_loader.load_activity(all_file, "Clean")
        elif (problem == "FordChallenge"):
            val_size = 10
            train_file = config['data_dir'] + '/' + problem + "_TRAIN.csv"
            test_file = config['data_dir'] + '/' + problem + "_TEST.csv"
            train_data = data_loader.load_ford_data(train_file, "Clean")
            test_data = data_loader.load_ford_data(test_file, "Clean")

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

        Data['max_len'] = window_len
        Data['train_data'] = X_train
        Data['train_label'] = y_train

        Data['val_data'] = X_val
        Data['val_label'] = y_val

        Data['test_data'] = X_test
        Data['test_label'] = y_test

        with open(config['data_dir'] + '/' + problem + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return Data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_runner(trainer, val_evaluator, epochs, path):
    total_start_time = time.time()
    tensorboard_writer = SummaryWriter('summary')
    best_value = 1e16
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
    save_best_model = utils.SaveBestModel()
    # save_best_acc_model = utils.SaveBestACCModel()

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc='Training Epoch', leave=False):

        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                              best_value, epoch)
        save_best_model(aggr_metrics_val['loss'], epoch, model, optimizer, loss_module, path)
        # save_best_acc_model(aggr_metrics_val['accuracy'], epoch, model, optimizer, loss_module, path)

        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
    total_runtime = time.time() - total_start_time
    logger.info("Train Time: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = Configuration().parse()  # "argsparse" object
    config = setup(args)  # configuration of Model Parameters
    device = Initialization(config)
    # Create Data ------------------------------------------------------------------------------------------------------
    if config['data_dir'][9:13] == 'Segm': # Loading HAR and Ford Data
        Data = load_segment_data(config)
        train_dataset = long_dataset_class(Data['train_data'], Data['train_label'])
        val_dataset = long_dataset_class(Data['val_data'], Data['val_label'])
        test_dataset = long_dataset_class(Data['test_data'], Data['test_label'])
    else:
        Data = load_UEA_data(config)  # Load UEA Data from *.ts data type
        train_dataset = dataset_class(Data['train_data'], Data['train_label'])
        val_dataset = dataset_class(Data['val_data'], Data['val_label'])
        test_dataset = dataset_class(Data['test_data'], Data['test_label'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                              collate_fn=lambda x: collate(x, max_len=Data['max_len']))
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                            collate_fn=lambda x: collate(x, max_len=Data['max_len']))
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                             collate_fn=lambda x: collate(x, max_len=Data['max_len']))
    # Create model ----------------------------------------------------------------------------------------------------
    logger.info("Creating model ...")
    model = model_factory(config, Data)
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))

    # Train Model -----------------------------------------------------------------------------------------------------
    optim_class = get_optimizer("RAdam")
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    loss_module = get_loss_module()
    save_path = os.path.join(config['save_dir'], 'model_{}.pth'.format('last'))
    tensorboard_writer = SummaryWriter('summary')
    model.to(device)
    # -----------------------------------------------------------------------------------------------------------------

    logger.info('Starting training...')
    time_start = time.perf_counter()

    trainer = SupervisedRunner(model, train_loader, device, loss_module, optimizer, l2_reg=0,
                               print_interval=config['print_interval'], console=config['console'], print_conf_mat=True)
    val_evaluator = SupervisedRunner(model, val_loader, device, loss_module, print_interval=config['print_interval'],
                                     console=config['console'], print_conf_mat=False)

    train_runner(trainer, val_evaluator, config["epochs"], save_path)

    best_model, optimizer, start_epoch = utils.load_model(model, save_path, optimizer)
    best_model.to(device)
    best_test_evaluator = SupervisedRunner(best_model, test_loader, device, loss_module,
                                           print_interval=config['print_interval'], console=config['console'],
                                           print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    time_elapsed = (time.perf_counter() - time_start)
    print(time_elapsed)
    print_str = 'Best Model Test Summary: '
    for k, v in best_aggr_metrics_test.items():
        print_str += '{}: {} | '.format(k, v)
    print(print_str)

    df_cm = pd.DataFrame(all_metrics["ConfMatrix"])
    df_cm.to_csv(os.path.join(config['save_dir'], 'Confmat.csv'))

    dic_results = {'Accuracy': all_metrics['total_accuracy'], 'Precision': all_metrics['precision'], 'Recall': all_metrics['recall'],
                   'Freq': all_metrics['support']}

    df_results = pd.DataFrame(dic_results)
    df_results.to_csv(os.path.join(config['save_dir'], 'Metrics.csv'))
    logger.info('All Done!')

