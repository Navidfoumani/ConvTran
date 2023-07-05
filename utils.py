import os
import json
import torch
import numpy as np
import logging
import zipfile
import requests
from datetime import datetime
from torch.utils.data import Dataset

from Dataset import load_segment_data, load_UEA_data

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def Setup(args):
    """
        Input:
            args: arguments object from argparse
        Returns:
            config: configuration dictionary
    """
    config = args.__dict__  # configuration dictionary
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, config['data_path'], initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}' as a configuration.json".format(output_dir))

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def Initialization(config):
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))
    return device


def Data_Loader(config):
    if config['data_dir'].split('/')[1] == 'Segmentation':
        Data = load_segment_data.load(config)  # Load HAR (WISDM V2) and Ford Datasets
    else:
        Data = load_UEA_data.load(config)  # Load UEA *.ts Data
    return Data


def Data_Verifier(config):

    if not os.path.exists(config['data_path']):
        os.makedirs(os.path.join(os.getcwd(), config['data_path']))
    directories = [name for name in os.listdir(config['data_path']) if os.path.isdir(os.path.join(config['data_path'], name))]

    if directories:
        print(f"The {config['data_path'].split('/')[1]} data is already existed")
    else:
        if config['data_path'].split('/')[1] == 'UEA':
            file_url = 'http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip'
            Downloader(file_url, 'UEA')

    if config['data_path'].split('/')[1] == 'UEA':
        config['data_path'] = os.path.join(config['data_path'], 'Multivariate_ts')


def Downloader(file_url, problem):
    # Define the path to download
    path_to_download = os.path.join('Dataset/', problem)
    # Send a GET request to download the file
    response = requests.get(file_url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Save the downloaded file
        file_path = os.path.join(path_to_download, 'Multivariate2018_ts.zip')
        with open(file_path, 'wb') as file:
            # Track the progress of the download
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 * 100  # 1KB
            downloaded_size = 0

            for data in response.iter_content(block_size):
                file.write(data)
                downloaded_size += len(data)

                # Calculate the download progress percentage
                progress = (downloaded_size / total_size) * 100

                # Print the progress message
                print(f' Download in progress: {progress:.2f}%')

        # Extract the contents of the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path_to_download)

        # Remove the downloaded zip file
        os.remove(file_path)

        print(f'{problem} Datasets downloaded and extracted successfully.')
    else:
        print(f'Failed to download the {problem} please update the file_url')
    return


class dataset_class(Dataset):

    def __init__(self, data, label):
        super(dataset_class, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, ind):

        x = self.feature[ind]
        x = x.astype(np.float32)

        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)



