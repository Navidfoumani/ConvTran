import glob
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sktime.utils import load_data
from einops import rearrange
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import random
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ts(root_dir):
    data_paths = glob.glob(os.path.join(root_dir, '*'))
    # Load Train Data
    if data_paths[1][-7:] == 'TEST.ts':
        train_path = data_paths[0]
        test_path = data_paths[1]
    else:
        train_path = data_paths[1]
        test_path = data_paths[0]

    train_df, train_labels = load_data.load_from_tsfile_to_dataframe(train_path, return_separate_X_and_y=True,
                                                         replace_missing_vals_with='NaN')
    labels = pd.Series(train_labels, dtype="category")
    # class_names = labels.cat.categories
    train_labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

    # Load Test Data
    test_df, test_labels = load_data.load_from_tsfile_to_dataframe(test_path, return_separate_X_and_y=True,
                                                                     replace_missing_vals_with='NaN')
    labels = pd.Series(test_labels, dtype="category")
    test_labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

    train_lengths = train_df.applymap(lambda x: len(x)).values
    test_lengths = test_df.applymap(lambda x: len(x)).values

    train_vert_diffs = np.abs(train_lengths - np.expand_dims(train_lengths[0, :], 0))

    if np.sum(train_vert_diffs) > 0:  # if any column (dimension) has varying length across samples
        train_max_seq_len = int(np.max(train_lengths[:, 0]))
        test_max_seq_len = int(np.max(test_lengths[:, 0]))
        max_seq_len = np.max([train_max_seq_len, test_max_seq_len])
        logger.warning("Not all samples have same length: maximum length set to {}".format(max_seq_len))
    else:
        max_seq_len = train_lengths[0, 0]
    # Replace NaN values
    grp = train_df.groupby(by=train_df.index)
    train_df = grp.transform(interpolate_missing)

    return train_df, train_labels_df, test_df, test_labels_df, max_seq_len


def collate(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels, IDs = zip(*data)
    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[2] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, 1, features[0].shape[1], max_len)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :, :, :end] = features[i]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks, IDs


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def split_dataset(data, label, validation_ratio):

    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data.loc[train_indices]
    train_label = label.loc[train_indices]
    val_data = data.loc[val_indices]
    val_label = label.loc[val_indices]
    return train_data.reset_index(drop=True), train_label.reset_index(drop=True),\
           val_data.reset_index(drop=True), val_label.reset_index(drop=True)


class dataset_class(Dataset):
    def __init__(self, df, label):
        super(dataset_class, self).__init__()

        lengths = df.applymap(lambda x: len(x)).values
        self.feature_df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0]*[row])) for row in range(df.shape[0])), axis=0)

        # feature = process_ts_data(df, normalise=False)


        # self.feature = feature.reshape(feature.shape[0], feature.shape[2], feature.shape[1])
        # self.mean = self.feature_df.mean()
        # self.std = self.feature_df.std()

        # self.feature_df = (self.feature_df - self.mean) / (self.std + np.finfo(float).eps)

        self.labels_df = label

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            y: (num_labels,) tensor of labels (num_labels > 1 for multi-task models) for each sample
            ID: ID of sample
        """

        X = self.feature_df.loc[ind].values  # (seq_length, feat_dim) array
        # X = self.feature[ind]
        X = rearrange(X, 'w d -> 1 d w')
        X = X.astype(np.float32)

        y = self.labels_df.loc[ind].values  # (num_labels,) array

        data = torch.tensor(X)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels_df)


class long_dataset_class(Dataset):
    def __init__(self, data, label):
        super(long_dataset_class, self).__init__()

        self.feature = data
        self.labels = label

    def __getitem__(self, ind):

        X = self.feature[ind]
        X = rearrange(X, 'w d -> 1 d w')
        X = X.astype(np.float32)

        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(X)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def process_ts_data(X, max_len,   vary_len: str = "suffix-noise",  normalise: bool = False):
    """
    This is a function to process the data, i.e. convert dataframe to numpy array
    :param X:
    :param normalise:
    :return:
    """
    num_instances, num_dim = X.shape
    columns = X.columns
    # max_len = np.max([len(X[columns[0]][i]) for i in range(num_instances)])
    output = np.zeros((num_instances, num_dim, max_len), dtype=np.float64)
    for i in range(num_dim):
        for j in range(num_instances):
            lengths = len(X[columns[i]][j].values)
            end = min(lengths, max_len)
            output[j, i, :end] = X[columns[i]][j].values
        output[:, i, :] = fill_missing(
            output[:, i, :],
            max_len,
            vary_len,
            normalise
        )

    d1, d2, d3 =output.shape
    return output.reshape(d1, d3, d2)


def fill_missing(x: np.array,
                 max_len: int,
                 vary_len: str = "suffix-noise",
                 normalise: bool = True):
    if vary_len == "zero":
        if normalise:
            x = StandardScaler().fit_transform(x)
        x = np.nan_to_num(x)
    elif vary_len == 'prefix-suffix-noise':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)
            diff_len = int(0.5 * (max_len - seq_len))

            for j in range(diff_len):
                x[i, j] = random.random() / 1000

            for j in range(diff_len, seq_len):
                x[i, j] = series[j - seq_len]

            for j in range(seq_len, max_len):
                x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    elif vary_len == 'uniform-scaling':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)

            for j in range(max_len):
                scaling_factor = int(j * seq_len / max_len)
                x[i, j] = series[scaling_factor]
            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    else:
        for i in range(len(x)):
            for j in range(len(x[i])):
                if np.isnan(x[i, j]):
                    x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]

    return x
