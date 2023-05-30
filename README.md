## Improving Position Encoding of Transformers for Multivariate Time Series Classification (ConvTran)
#### Authors: [Navid Mohammadi Foumani](https://www.linkedin.com/in/navid-foumani/), [Chang Wei Tan](https://changweitan.com/), [Geoffrey I. Webb](https://i.giwebb.com/), [Mahsa Salehi](https://research.monash.edu/en/persons/mahsa-salehi)

#### ConvTran Paper: [Preprint](https://arxive.org/)
This is a PyTorch implementation of Improving Position Encoding of Transformers for Multivariate Time Series Classification (ConvTran)
![img](https://github.com/Navidfoumani/ConvTran/blob/7d77755f59a596b9a62f0aa3ae75fef1edd7d4f2/Fig/ConvTran.png)

## Overview 
<p align="justify">
Attention models have the exceptional ability to capture long-range dependencies and their broader receptive fields provide more contextual information, which can improve the models’ learning capacity. However, these models have a limitation when capturing the order of input series. Hence, adding explicit representations of position information is especially important for attention since the model is otherwise entirely invariant to input order, which is undesirable for modeling sequential data. This limitation is particularly challenging in the case of time series data, where the absence of Word2Vec-like embeddings diminishes the availability of informative contextual cues.
</p>

<p align="justify">
The question of whether absolute position encoding, relative position encoding, or a combination of both is more suitable for capturing the sequential nature of time series data remains unresolved. To bridge this gap, our paper reviews existing absolute and relative position encoding methods applied in time series classification. Additionally, we propose two novel position encoding techniques: time Absolute Position Encoding (tAPE) and efficient Relative Position Encoding (eRPE).
</p>

### Key Idea of Position Encoding of Transformers for MTSC
#### time Absolute Position Encoding (tAPE)
<p align="justify">
The original proposal of absolute position encoding was primarily intended for language modeling, where position embeddings with high dimensions, such as 512 or 1024, are commonly utilized for inputs of length 512. The figure below illustrates the dot product between two sinusoidal positional embeddings at a distance of K, using different embedding dimensions.
</p>

<p align="justify">
It is evident that higher embedding dimensions, such as 512 (indicated by the red thick line), provide a more accurate reflection of the similarity between different positions. Conversely, when employing lower embedding dimensions, such as 64 or 128 (represented by the thin blue and orange lines, respectively), the dot product does not consistently decrease as the distance between positions increases. This phenomenon, known as the "distance awareness" property, is present in higher dimensions but diminishes with lower embedding dimensions, such as 64.
</p>
<pre>
<code>
# Equation 13 page 11
class tAPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(timeAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)
</code>
</pre>
<p align="justify">
Without our modification, as series length $L$ increases the dot product of positions becomes ever less regular, resulting in a loss of distance awareness. By incorporating the length parameter in the frequency terms in both sine and cosine functions, the dot product remains smoother with a monotonous trend. As the embedding dimension $d_{model}$ value increases, it is more likely the vector embeddings are sampled from low-frequency sinusoidal functions, which results in the anisotropic phenomenon. To alleviate this, we incorporate the $d_{model}$ parameter into the frequency term in both sine and cosine functions
</p>

#### efficient Relative Position Encoding (eRPE)


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
