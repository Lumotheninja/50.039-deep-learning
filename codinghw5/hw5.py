from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

class LanguageDataset(torch.utils.data.Dataset):

    def __init__(self, data, label):
        self.label_list = torch.tensor(label)
        self.tensor_list = [self.lineToTensor(line).squeeze() for line in data]
        # self.tensor_list = torch.nn.utils.rnn.pad_sequence([self.lineToTensor(line).squeeze() for line in data], batch_first=True)

    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        return self.tensor_list[idx], self.label_list[idx]

    def getInputSize(self):
        return self.tensor_list[0].size()[-1]
    
    def letterToIndex(self, letter):
        return all_letters.find(letter)

    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, class_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.class_size = class_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, class_size)
        #self.fc = torch.nn.Linear(layer_size*hidden_size, class_size)

    def forward(self, input):
        num_batch, _, __ = input.shape
        input = torch.nn.utils.rnn.pack_padded_sequence(input, [v.size(0) for v in input], batch_first=True)
        h0 = torch.zeros(self.layer_size, num_batch, self.hidden_size)
        c0 = torch.zeros(self.layer_size, num_batch, self.hidden_size)
        output, (hidden, cell)= self.lstm(input, (h0, c0))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return self.fc(output[:, -1, :])
        # return (self.fc(hidden.view(num_batch, self.layer_size * self.hidden_size)))

def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def train(model, loss_fn, device, train_loader, optimizer):
    model.train()
    samples_trained = 0
    training_loss = 0
    for data, target in train_loader:
        correct_pred = 0
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        training_loss += loss.item()
        pred = output.argmax(dim=1)
        correct_pred += torch.sum(pred==target).item()
        loss.backward()
        optimizer.step()
        samples_trained += data.size()[0]
        print('Trained %d samples, loss: %10f' %(samples_trained, training_loss/samples_trained), end="\r")
    training_loss /= samples_trained
    return training_loss

def test(model, loss_fn, device, test_loader):
    model.eval()
    correct_pred = 0
    samples_tested = 0
    test_loss = 0
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct_pred += torch.sum(pred==target).item()
            samples_tested += data.size()[0]
            #print (pred)
            print('Tested %d samples, loss: %10f' %(samples_tested, test_loss/samples_tested), end="\r")
    accuracy = correct_pred/samples_tested
    test_loss /= samples_tested
    return accuracy, test_loss

def train_test_test(model, loss_fn, device, optimizer, epoch, name):

    training_loss_list = []
    test_loss_list = []
    test_accuracy_list = []
    highest_test_loss = 1

    # training and testidating across epochs
    for t in range(epoch):
        print ('Current epoch: ', t+1)
        training_loss = train(model, loss_fn, device, train_loader, optimizer)
        print ('')
        accuracy, test_loss = test(model, loss_fn, device, test_loader)
        print ('')
        if test_loss < highest_test_loss:
            highest_test_loss = test_loss
            torch.save(model.state_dict(), name + '.pt')
        test_accuracy_list.append(accuracy)
        print ("Accuracy", accuracy)
        training_loss_list.append(training_loss)
        test_loss_list.append(test_loss)

    # plot and save graph
    plt.plot(test_loss_list, label='test loss')
    plt.legend(loc='upper left')
    plt.savefig(name + '_test_loss.png')
    plt.clf()

    plt.plot(training_loss_list, label='training loss')
    plt.legend(loc='upper left')
    plt.savefig(name + '_training_loss.png')
    plt.clf()

    plt.plot(test_accuracy_list, label='test accuracy')
    plt.legend(loc='upper left')
    plt.savefig(name + '_test_acc.png')
    plt.clf()

    # testing
    '''
    model.load_state_dict(torch.load(name + '.pt'))
    test_accuracy, _ = test(model, loss_fn, device, test_loader)
    print ("Accuracy is: ", test_accuracy)
    with open(name + '.txt', 'w') as txtfile:
        txtfile.write("Accuracy is: "+ str(test_accuracy))
    '''

def collate_fn(batch):
    data = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True)
    target = torch.tensor([item[1] for item in batch])
    #print (data.size())
    return data, target

test_split = .2
random_seed= 24
batch_size =  30
category_list = []
train_list = []
test_list = []
train_label_list = []
test_label_list = []
n_classes = 0

for idx, filename in enumerate(glob.glob('data/names/*.txt')):
    n_classes += 1
    category = os.path.splitext(os.path.basename(filename))[0]
    category_list.append(category)
    lines = readLines(filename)
    split = int(np.floor(test_split * len(lines)))
    np.random.seed(random_seed)
    np.random.shuffle(lines)
    train_list.extend(lines[split:])
    test_list.extend(lines[:split])
    train_label_list.extend(idx for _ in range(len(lines[split:])))
    test_label_list.extend(idx for _ in range(len(lines[:split])))

print ("Total train data: ", len(train_list))
print ("Total test data", len(test_list))
print ("Total num classes", n_classes)

train_dataset = LanguageDataset(train_list, train_label_list)
test_dataset = LanguageDataset(test_list, test_label_list)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

n_hidden = 50
n_layer= 2

rnn = RNN(train_dataset.getInputSize(), n_hidden, n_layer, n_classes)
epoch = 10
loss = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer=torch.optim.Adam(rnn.parameters(),lr=learning_rate)
device = torch.device('cpu')
train_test_test(rnn, loss, device, optimizer, epoch, 'batchsize30_h50')
