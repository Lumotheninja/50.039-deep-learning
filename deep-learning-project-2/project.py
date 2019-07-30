from io import open, StringIO
import glob
import os
import string
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
from unidecode import unidecode


all_letters = string.printable #^ is EOS
n_letters = len(all_letters)

def get_data():
    title_lines = []
    content_lines = []
    title_start_letters = []
    content_start_letters = []
    with open('smaller.csv', newline='\n') as f:
        reader = csv.reader(f, quotechar='\'')
        for row in reader:
            if (len(row) > 0):
                title = unidecode(row[0]).replace('"', '\'')
                # content = unidecode(row[1]).replace('"', '\'')
                if len(title.split(' ')) > 30:
                    continue
                title_lines.append(title + '^')
                title_start_letters.append(title[0])
                # content_lines.append(content + '^')
                # content_start_letters.append(content[0])
    return title_lines, content_lines, title_start_letters, content_start_letters

def letterToIndex(letter):
    if all_letters.find(letter) == -1:
        print ("char %s not found in vocab" %letter) #find missing letters
    return all_letters.find(letter)

def indexToLetter(idx):
    return all_letters[idx]

def lineToTensor(line):
    tensor = torch.zeros(len(line), n_letters)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor

title_lines, content_lines, title_start_letters, content_start_letters = get_data()

class LanguageDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.char_list = [[letterToIndex(char) for char in line] for line in data]
        self.tensor_list = [lineToTensor(line) for line in data]

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return self.tensor_list[idx], self.char_list[idx]

    def getInputSize(self):
        return self.tensor_list[0].size()[-1]
    
    

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, device, start_letters):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = input_size # input = output
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.device = device
        self.start_letters = start_letters
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer_size, batch_first=True, dropout=0.1)
        self.fc = torch.nn.Linear(hidden_size, self.output_size)

    def forward(self, input):
        num_batch, _, __ = input.shape
        h0 = torch.zeros(self.layer_size, num_batch, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.layer_size, num_batch, self.hidden_size).to(self.device)
        output, _ = self.lstm(input, (h0, c0))
        return self.fc((output.squeeze()))

    def sample(self, temperature):
        self.eval()
        curr_letter = np.random.choice(self.start_letters) #start with uppercase
        sentence = ''
        h = torch.zeros(self.layer_size, 1, self.hidden_size).to(self.device)
        c = torch.zeros(self.layer_size, 1, self.hidden_size).to(self.device)
        with torch.no_grad():
            while curr_letter != '^':
                sentence += curr_letter
                t = torch.zeros(1, 1, n_letters).to(self.device)
                t[0][0][letterToIndex(curr_letter)] = 1
                output, (h, c) = self.lstm(t, (h, c))
                output = self.fc(output.squeeze()).squeeze() 
                probs = torch.nn.functional.softmax(output/temperature, dim=0).cpu().numpy()
                curr_idx = np.random.choice(np.arange(n_letters), p=probs)
                curr_letter = indexToLetter(curr_idx)
        return sentence


def train(model, loss_fn, device, train_loader, optimizer):
    model.train()
    samples_trained = 0
    training_loss = 0
    for data, char in train_loader:
        correct_pred = 0
        data, char = data.to(device).unsqueeze(0), char.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, char)
        training_loss += loss.item()
        pred = output.argmax(dim=1)
        correct_pred += torch.sum(pred==char).item()
        loss.backward()
        optimizer.step()
        samples_trained += data.size()[0]
        print('Trained %d samples, loss: %10f' %(samples_trained, training_loss/samples_trained), end="\r")
    training_loss /= samples_trained
    return training_loss

def test(model, loss_fn, device, test_loader):
    model.eval()
    accuracy = 0
    samples_tested = 0
    test_loss = 0
    with torch.no_grad():
        for data, char in test_loader:
            data, char = data.to(device).unsqueeze(0), char.to(device)
            output = model(data)
            test_loss += loss_fn(output, char).item()
            pred = output.argmax(dim=1)
            #print (''.join([indexToLetter(char) for char in pred.cpu().numpy()])) 
            accuracy += torch.sum(pred==char).item()/(data.size()[1]-1)
            samples_tested += data.size()[0]
            print('Tested %d samples, loss: %10f' %(samples_tested, test_loss/samples_tested), end="\r")
    accuracy /= (samples_tested)
    test_loss /= (samples_tested)
    return accuracy, test_loss

def train_test_generate(model, loss_fn, device, optimizer, epoch, temperature, train_loader, test_loader, name):

    training_loss_list = []
    test_loss_list = []
    test_accuracy_list = []
    highest_test_loss = 100

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
        with open ('output%d.txt' %epoch, 'w') as file:
            for i in range(10):
                text = model.sample(temperature)
                print (text)
                file.write(text) 
                file.write('\n')
                file.write('\n')

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


def collate_fn(batch):
    #data = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True)
    data = batch[0][0][:-1] # no need EOS
    target = torch.Tensor(batch[0][1][1:]).long()
    return data, target

isdev = False
test_split = .2
random_seed= 12
batch_size = 1

title_lines, content_lines, title_start_letters, content_start_letters = get_data()
if isdev:
    title_lines = title_lines[:100]
# title_and_content = list(zip(title_lines, content_lines))
# np.random.shuffle(title_and_content)
# title_lines, content_lines = zip(*title_and_content)
split = int(np.floor(test_split * len(title_lines)))

title_train_list = title_lines[split:]
title_test_list = title_lines[:split]
# content_train_list = content_lines[split:]
# content_test_list = title_lines[:split]

title_train_dataset = LanguageDataset(title_train_list)
title_test_dataset = LanguageDataset(title_test_list)
# content_train_dataset = LanguageDataset(content_train_list)
# content_test_dataset = LanguageDataset(content_test_list)

title_train_loader = torch.utils.data.DataLoader(title_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
title_test_loader = torch.utils.data.DataLoader(title_test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# content_train_loader = torch.utils.data.DataLoader(content_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# content_test_loader = torch.utils.data.DataLoader(content_test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

print ("Total train data: ", len(title_train_list))
print ("Total test data", len(title_test_list))

n_hidden = 200
n_layer= 3
temp = 0.5

device = torch.device('cpu') if isdev else torch.device('cuda:0')
rnn = RNN(title_train_dataset.getInputSize(), n_hidden, n_layer, device, title_start_letters).to(device)
epoch = 10 if isdev else 30
loss = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer=torch.optim.Adam(rnn.parameters(), lr=learning_rate)


train_test_generate(rnn, loss, device, optimizer, epoch, temp, title_train_loader, title_test_loader, 'model')
