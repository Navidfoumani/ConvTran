## ConvTran: Improving Position Encoding of Transformers for Multivariate Time Series Classification
#### Authors: [Navid Mohammadi Foumani](https://www.linkedin.com/in/navid-foumani/), [Chang Wei Tan](https://changweitan.com/), [Geoffrey I. Webb](https://i.giwebb.com/), [Mahsa Salehi](https://research.monash.edu/en/persons/mahsa-salehi)

#### ConvTran Paper: [Preprint](https://arxive.org/)
## Overview 
This is a PyTorch implementation of ConvTran : Deep Learning for Multivariate Time Series Classification Through Tight Integration of Convolutions and Transformers
![img](https://github.com/Navidfoumani/ConvTran/blob/e41fb4b387ec5c2351df4416fdd326dda7801a1c/Fig/ConvTran.png)

### Get data from UEA Archive and HAR and Ford Challenge
Download dataset files and place them into the specified folder
UEA: http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip
Copy the datasets folder to: Datasets/UEA/

HAR : https://www.cis.fordham.edu/wisdm/dataset.php
Copy the ActivityRecognition.txt file to : Datasets/Segmentation/ActivityRecognition/

Ford: https://www.kaggle.com/competitions/stayalert/data
Copy the FordChallenge_TEST.csv and FordChallenge_Train.csv to : Datasets/Segmentation/FordChallenge

## Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

This code has been tested with `Python 3.7` and `3.8`.

`pip install -r requirements.txt`

## Run

To see all command options with explanations, run: `python main.py --help`
In 'configuration.py' you can select the datasets and modify the model parameters.
For example:

`self.parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')`

or you can set the paprameters:

`python main.py --epochs 1500 --data_dir Datasets/UEA/Heartbeat`

## Credits

Some Part of Code are taken from [TST](https://github.com/gzerveas/mvts_transformer).
