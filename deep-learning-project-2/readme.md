# You won't believe what these SUTD students do for their deep learning project!
In this report, we attempt to generate clickbait titles using a recurrent neural network, given an article. We decided to use a encoder-decoder model to generate the titles. The decoder works to generate the titles while the encoder creates the hidden state for the decoder to generate the title.

# Data
We have 2 sets of data, one with just clickbait titles for training the decoder and one with clickbait titles and articles for training the encoder.

## Dataset 1: 
We combined 700 data points from https://www.clickbait-challenge.org/#data and 16000+ clickbait titles from https://github.com/bhargaviparanjape/clickbait/tree/master/dataset

## Dataset 2: 
We extracted the clickbait titles from https://github.com/several27/FakeNewsCorpus which has 200,000+ clickbait titles and articles

For embeddings, FastText word embeddings trained on Simple English Wikipedia articles were obtained from https://fasttext.cc/docs/en/pretrained-vectors.html

# Model 1
## Pretraining the weights on dataset 1
Weights were first pre-trained on dataset 1, and transfer learning was applied to retrain the model on dataset 2.

## Preprocessing
All punctuation was removed from the text. UTF-8 strings were converted into ascii strings to reduce the vocabulary size when training the decoder, and only the first 10 000 articles were used as the full dataset occupies more than 69 GB RAM when converted to tensors.

## Architecture
| Hyperparameter          | Encoder | Decoder                 |
|-------------------------|---------|-------------------------|
| Hidden Dimensions       | 200     | 200                     |
| Number of layers        | 3       | 3                       |
| Dropout                 | None    | 0.1                     |
| Optimizer               | Adam    | Adam                    |
| Learning rate           | 0.001   | 0.001                   |
| Batch size              | 1       | 1                       |
| Number of epochs        | 30      | 30                      |
| Sampling temeperature   | -       | 0.5                     |
| Start of sequence input | None    | Start of Sentence token |
| Loss                    | -       | Cross Entropy Loss      |


# Model 2
## Preprocessing
FastText Word Embeddings were used to embedded articles, while one-hot character vectors were used to embed titles. Due to the large data size, only articles with less than 10 Out-Of-Vocabulary (OOV) words were used, with a final selection of round 60 000 articles. OOV words and common stop words such as ‘and’ were excluded from embeddings in order to speed up the model, as they are expected to contribute little in helping the model to grasp the article’s content.

## Architecture
| Hyperparameter          | Encoder | Decoder     |
|-------------------------|---------|-------------|
| Hidden Dimensions       | 300     | 300         |
| Number of layers        | 2       | 2           |
| Dropout                 | 0.1     | 0.1         |
| Optimizer               | Adam    | Adam        |
| Learning rate           | 0.001   | 0.001       |
| Batch size              | 4       | 4           |
| Number of epochs        | 10      | 10          |
| Sampling temeperature   | -       | 0.5         |
| Start of sequence input | None    | Zero-vector |
| Loss                    | -       | NLL Loss    |

# Model evaluation
For model 2, the decision to exclude OOV words which are difficult to accurately embed from the article was a poor one, as those words occur in the title, and the model randomly inserts common OOV words into the title even if they do not appear in the article - such as Trump and Clinton - since that was what it was trained on.  

Although different loss functions are used, they are essentially the same formula since the probability of the ground truth character is 1. The lower test loss for the second model is likely attributable to the number of hidden dimensions in the LSTM. As most word encoding models have at least 300 dimensions, it can be inferred that at least 300 dimensions is necessary to mathematically express the semantics of words or sentences. 
A second possible source of the better performance is the usage of larger datasets than the first model, despite transfer learning being applied only to the first model, which has allowed it to generalize better. This is evident from the training and testing graph of the first model, where we can see overfitting occuring after 5 epochs, as compared to no overfitting for the second model.
