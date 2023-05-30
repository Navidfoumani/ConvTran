import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

__author__ = "Chang Wei Tan and Navid Foumani"


def load_segmentation_data(file_path, data_type, norm=True, verbose=1):
    # Load the data from csv. Assumes that the data is in dataframe format.
    # Data has columns for:
    #   series:     the id for each series in the dataset
    #   label:      the running labels for every timepoint
    #   timestamp:  timestamp of the data
    #   d1-dN:      time series data with N dimensions
    #
    # Read the data in that format and store it into a new format with data and label columns.
    # The data is an array of shape = seq_len, n_dim
    if verbose > 0:
        print("[Data_Loader] Loading data from {}".format(file_path))

    df = pd.read_csv(file_path)
    drive = [3, 11]
    if data_type == "Clean":
        # Drop other class data ------------------------------------------------------------------------------
        Other_Class = [0, 1, 2, 12]  # "X", "EyesCLOSEDneutral", "EyesOPENneutral", "LateBoredomLap"
        df = df.drop(np.squeeze(np.where(np.isin(df['label'], Other_Class))))
        distract = [4, 5, 6, 7, 8, 9, 10, 13, 14, 15]
        # -----------------------------------------------------------------------------------------------------
    else:
        distract = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
    all_series = df.series.unique()
    data = []
    for series in all_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)

        this_series.label = this_series.label.replace(distract, 0)
        this_series.label = this_series.label.replace(drive, 1)

        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        data.append(pd.DataFrame({"data": [series_data],
                                  "label": [series_labels]}, index=[0]))
    data = pd.concat(data)
    data.reset_index(drop=True, inplace=True)
    return data


def load_activity(file_path, data_type, norm=True, verbose=1):

    column_names = ['series', 'label', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv(file_path, header=None, names=column_names, comment=';')
    df.dropna(axis=0, how='any', inplace=True)
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['label'])
    all_series = df.series.unique()
    train_series, test_series = train_test_split([x for x in range(len(all_series))], test_size=6, random_state=1)
    train_data = []
    print("[Data_Loader] Loading Train Data")
    for series in train_series:
        series = series + 1
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        train_data.append(pd.DataFrame({"data": [series_data],
                                        "label": [series_labels]}, index=[0]))
    train_data = pd.concat(train_data)
    train_data.reset_index(drop=True, inplace=True)
    test_data = []
    print("[Data_Loader] Loading Test Data")
    for series in test_series:
        series = series + 1
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        test_data.append(pd.DataFrame({"data": [series_data],
                                       "label": [series_labels]}, index=[0]))
    test_data = pd.concat(test_data)
    test_data.reset_index(drop=True, inplace=True)
    return train_data, test_data


def load_ford_data(file_path, data_type, norm=True, verbose=1):
    if verbose > 0:
        print("[Data_Loader] Loading data from {}".format(file_path))
    df = pd.read_csv(file_path)
    all_series = df.series.unique()
    data = []

    for series in all_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        data.append(pd.DataFrame({"data": [series_data],
                                  "label": [series_labels]}, index=[0]))
    data = pd.concat(data)
    data.reset_index(drop=True, inplace=True)

    return data