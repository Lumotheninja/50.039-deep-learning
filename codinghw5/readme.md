# LSTM with name dataset

Trained [Pytorch LSTM](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py) on the country names dataset, based on the [excellent Pytorch character level-rnn tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
To let LSTM train on variable sequences, we use rnn pad sequence which pads the sequence with 0 tensors, we run this for every batch size that we want to input into the LSTM. Hence each batch might have different sequence length.  

# Results
Test accuracy with 3 different transforms:  
1. LSTM with batchsize of 1: 0.775
2. LSTM with batchsize of 10: 0.467
3. LSTM with batchsize of 30: 0.467
The 2 accuracies which are 0.467 are obtained by predicting all class 14 for the name problem, this is due to the padding per batch affecting the hidden state of the LSTM.

# Model1 Graphs:
1. ![training loss vs epoch](./results/model1_training_loss.png)
2. ![test loss vs epoch](./results/model1_test_loss.png)
3. ![test accuracy vs epoch](./results/model1_test_acc.png)

# Model1 Graphs:
1. ![training loss vs epoch](./results/model2_training_loss.png)
2. ![test loss vs epoch](./results/model2_test_loss.png)
3. ![test accuracy vs epoch](./results/model2_test_acc.png)

# Model3 Graphs:
1. ![training loss vs epoch](./results/model3_training_loss.png)
2. ![test loss vs epoch](./results/model3_test_loss.png)
3. ![test accuracy vs epoch](./results/model3_test_acc.png)