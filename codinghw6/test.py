from io import open
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
import re
from io import StringIO

all_letters = all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
n_letters = len(all_letters)

def get_data():
    category_lines = {}
    start_letters = []
    all_categories = ['st']
    category_lines['st'] = []
    filterwords=['NEXTEPISODE']
    with open('./data/star_trek_transcripts_all_episodes_f.csv', newline='') as f:
        io = StringIO(re.sub(r',[^\s-]', '\t', f.read()))
        reader = csv.reader(io, delimiter='\t', quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el)>1):
                    v = re.sub(r'[;\"=/]', '', el)
                    category_lines['st'].append(v)
                    start_letters.append(v[0])
    n_categories = len(all_categories)
    return category_lines, all_categories, start_letters

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
        self.output_size = input_size + 1# input = output
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.device = device
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        # self.fc = torch.nn.Linear(hidden_size, class_size)
        self.fc = torch.nn.Linear(hidden_size, self.output_size)

    def forward(self, input):
        num_batch, _, __ = input.shape
        #input = torch.nn.utils.rnn.pack_padded_sequence(input, [v.size(0) for v in input], batch_first=True)
        h0 = torch.zeros(self.layer_size, num_batch, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.layer_size, num_batch, self.hidden_size).to(self.device)
        output, _ = self.lstm(input, (h0, c0))
        #output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return self.fc(self.dropout(output.squeeze()))
        # return (self.fc(hidden.view(num_batch, self.layer_size * self.hidden_size)))

    def sample(self, temperature):
        self.eval()
        curr_idx = letterToIndex(np.random.choice(start_letters)) #start with uppercase
        sentence = ''
        h = torch.zeros(self.layer_size, 1, self.hidden_size).to(self.device)
        c = torch.zeros(self.layer_size, 1, self.hidden_size).to(self.device)
        with torch.no_grad():
            while curr_idx != n_letters:
                sentence += indexToLetter(curr_idx)
                t = torch.zeros(1, 1, n_letters).to(self.device)
                t[0][0][curr_idx] = 1
                output, (h, c) = self.lstm(t, (h, c)) 
                output = self.fc(self.dropout(output.squeeze())).squeeze(0).squeeze(0)
                prob = torch.softmax(output/temperature, 0).detach().cpu().numpy()
                curr_idx = np.random.choice(np.arange(n_letters + 1), p=prob)
        return sentence

test_split = .2

lines, category, start_letters = get_data()
lines = lines['st']
np.random.shuffle(lines)
split = int(np.floor(test_split * len(lines)))
train_list = lines[split:]
test_list = lines[:split]

train_dataset = LanguageDataset(train_list)
n_hidden = 200
n_layer= 3
temp = 0.5

device = 'cpu'
print (train_dataset.getInputSize())
rnn = RNN(train_dataset.getInputSize(), n_hidden, n_layer, device, start_letters).to(device)
rnn.load_state_dict(torch.load('test.pt', map_location='cpu'))
print ('here we go')
print (rnn.sample(0.5))
