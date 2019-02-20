from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt


# model definition

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# training 

def train(model, device, train_loader, optimizer):
    model.train()
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = torch.reshape(data,(data.shape[0],784)).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum') #
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
    training_loss /= len(train_loader.dataset)
    
    print('Training set: Average loss: {:.4f}'.format(training_loss))
    
    return training_loss

#testing 

def test(model, device, test_loader, oldparams, oldloss):
    model.eval()
    test_loss = 0
    correct = 0
    correct_list = [0 for w in range(10)]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = torch.reshape(data,(data.shape[0],784)).to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for t, p in zip(target.view(-1), pred.view(-1)):
                if t.data==p.data:
                    correct_list[t] += 1 
    test_loss /= len(test_loader.dataset)
    
    
    print ('Test set class-wise accuracy: ', [100*x/(len(test_loader.dataset)/10) for x in correct_list])
        
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if test_loss < oldloss:
        return test_loss, model.state_dict(), test_loss
    else:
        return test_loss,oldparams, oldloss

def run():

    # first we set our constants
    epochs = 100
    learningrate = 0.01
    device = torch.device('cpu')
    model = NN()
    optimizer=torch.optim.SGD(model.parameters(),lr=learningrate, momentum=0.0, weight_decay=0)
    
    # load data from FashionMNIST
    xy_train = FashionMNIST('', train = True, transform = transforms.ToTensor())
    xy_test = FashionMNIST('', train = False, transform = transforms.ToTensor())

    #data loader class
    loadertr=torch.utils.data.DataLoader(xy_train,batch_size=60,shuffle=True)
    loadertest=torch.utils.data.DataLoader(xy_test,batch_size=10,shuffle=False) 
    
    # initialize oldparams, oldloss
    oldparams = model.state_dict()
    oldloss = float('inf')

    # array for plotting
    training_loss_array = []
    test_loss_array = []

    for x in range(epochs):
        print ('current epoch:', x+1)
        training_loss = train(model, device, loadertr, optimizer)
        test_loss, oldparams, oldloss = test(model, device, loadertest, oldparams, oldloss)
        training_loss_array.append(training_loss)
        test_loss_array.append(test_loss)

    plt.plot(training_loss_array, label='training loss')
    plt.plot(test_loss_array, label='test loss')
    plt.legend(loc='upper left')
    plt.show()
    torch.save(model.state_dict(), 'model_state.json')

if __name__=='__main__':
    run()