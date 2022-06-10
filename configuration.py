import argparse
import logging
import os
import json
import torch
from datetime import datetime
from utils import utils

logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


# Adopted from https://github.com/gzerveas/mvts_transformer
class Configuration(object):

    def __init__(self):
        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description='Run a complete training. JSON configuration file can be used to overwrite command-line')
        self.parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')

        # For UEA datasets : default='Datasets/UEA/InsectWingbeat' , default='Datasets/UEA/Heartbeat'
        # For Ford datasets: default='Datasets/Segmentation/FordChallenge
        self.parser.add_argument('--data_dir', default='Datasets/UEA/Heartbeat', help='Data directory')

        self.parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of validation set")

        # Model ---------------------------------------------------------------------------------------
        self.parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')

        self.parser.add_argument('--d_model', type=int, default=128, help='Internal dimension of transformer embeddings')

        self.parser.add_argument('--dim_ff', type=int, default=128, help='Dimension of feedforward part of transformer layer')

        self.parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')

        self.parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')

        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

        self.parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation set')

        self.parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                                 help='Metric used for defining best epoch')

        # I/O -------------------------------------------------------------------------------------------------------
        self.parser.add_argument('--output_dir', default='Results', help='Root output directory. Must exist.')

        # System
        self.parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
        self.parser.add_argument('--console', action='store_true',
                                 help="Optimize printout for console output; otherwise for file")

        self.parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')

    def parse(self):
        args = self.parser.parse_args()
        return args


# Adopted from https://github.com/gzerveas/mvts_transformer
def setup(args):

        '''
        Prepare training session: read configuration from file (takes precedence), create directories.
        Input:
            args: arguments object from argparse
        Returns:
            config: configuration dictionary
        '''

        config = args.__dict__  # configuration dictionary
        '''
        if args.config_filepath is not None:
            logger.info("Reading configuration ...")
            try:  # dictionary containing the entire configuration settings in a hierarchical fashion
                config.update(utils.load_config(args.config_filepath))
            except:
                logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
                traceback.print_exc()
                sys.exit(1)
        '''
        # Create output directory
        initial_timestamp = datetime.now()
        output_dir = config['output_dir']
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_dir = os.path.join(output_dir, config['data_dir'], initial_timestamp.strftime("%Y-%m-%d_%H-%M"))

        config['output_dir'] = output_dir
        config['save_dir'] = os.path.join(output_dir, 'checkpoints')
        config['pred_dir'] = os.path.join(output_dir, 'predictions')
        config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
        utils.create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

        # Save configuration as a (pretty) json file
        with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
            json.dump(config, fp, indent=4, sort_keys=True)

        logger.info("Stored configuration file in '{}'".format(output_dir))

        return config


def Initialization(config):
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))
    return device
