import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt

isdev = False

class FlowersDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, dataset_file, transform):
        self.img_dir = img_dir
        self.dataset_file = dataset_file
        self.transform = transform

    def __len__(self):
        with open(self.dataset_file) as file:
            lines = file.readlines()
        return len(lines)
        #return 50 if isdev else len(lines)
    
    def __getitem__(self, idx):
        with open(self.dataset_file) as file:
            img_name, label = file.readlines()[idx].strip('\n').split(' ')
            img_name = os.path.join(self.img_dir, img_name)
            img = self.transform(Image.open(img_name))
            label = torch.tensor(float(label)).long()
        return img, label

def train(model, loss_fn, device, train_loader, optimizer):
    model.train()
    samples_trained = 0
    training_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
        samples_trained += data.size()[0]
        print('Trained %d samples, loss: %10f' %(samples_trained, training_loss/samples_trained), end="\r")
        del data
        del target
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
            print('Tested %d samples, loss: %10f' %(samples_tested, test_loss/samples_tested), end="\r")
            del data
            del target
    accuracy = correct_pred/samples_tested
    test_loss /= samples_tested
    return accuracy, test_loss
        
def train_val_test(model, loss_fn, device, optimizer, epoch, name):

    training_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    highest_val_loss = 1

    # training and validating across epochs
    for t in range(epoch):
        print ('Current epoch: ', t+1)
        training_loss = train(model, loss_fn, device, train_loader, optimizer)
        print ('')
        accuracy, val_loss = test(model, loss_fn, device, val_loader)
        print ('')
        if val_loss < highest_val_loss:
            highest_val_loss = val_loss
            torch.save(model.state_dict(), name + '.pt')
        val_accuracy_list.append(accuracy)
        training_loss_list.append(training_loss)
        val_loss_list.append(val_loss)

    # plot and save graph
    plt.plot(val_loss_list, label='val loss')
    plt.legend(loc='upper left')
    plt.savefig(name + '_val_loss.png')
    plt.clf()

    plt.plot(training_loss_list, label='training loss')
    plt.legend(loc='upper left')
    plt.savefig(name + '_training_loss.png')
    plt.clf()

    plt.plot(val_accuracy_list, label='val accuracy')
    plt.legend(loc='upper left')
    plt.savefig(name + '.png')
    plt.clf()

    # testing
    model.load_state_dict(torch.load(name + '.pt'))
    test_accuracy, _ = test(model, loss_fn, device, test_loader)
    print ("Accuracy is: ", test_accuracy)
    with open(name + '.txt', 'w') as txtfile:
        txtfile.write("Accuracy is: "+ str(test_accuracy))

if __name__ == '__main__':
    train_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                                   'flowersstuff/trainfile.txt', 
                                   transforms.Compose([transforms.Resize(size=224), 
                                                        transforms.CenterCrop(224), 
                                                        transforms.ToTensor(), 
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    val_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                                   'flowersstuff/valfile.txt', 
                                   transforms.Compose([transforms.Resize(size=224), 
                                                        transforms.CenterCrop(224), 
                                                        transforms.ToTensor(), 
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    test_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                                   'flowersstuff/testfile.txt', 
                                   transforms.Compose([transforms.Resize(size=224), 
                                                        transforms.CenterCrop(224), 
                                                        transforms.ToTensor(), 
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=1)
    
    
    # model1
    device = torch.device('cuda:0')
    loss_fn = F.cross_entropy
    epoch = 5 if isdev else 30
    learningrate=0.1
    
    model1 = models.resnet18(pretrained=False)
    model1.fc = torch.nn.Linear(512, 102)
    model1.to(device)
    optimizer1=torch.optim.SGD(model1.parameters(),lr=learningrate)
    train_val_test(model1, loss_fn, device, optimizer1, epoch, 'model1')
    del model1

    # model2
    model2 = models.resnet18(pretrained=True)
    model2.fc = torch.nn.Linear(512, 102)
    model2.to(device)
    optimizer2=torch.optim.SGD(model2.parameters(),lr=learningrate)
    train_val_test(model2, loss_fn, device, optimizer2, epoch, 'model2')
    del model2
    

    # model3
    model3 = models.resnet18(pretrained=True)
    model3.fc = torch.nn.Linear(512, 102)
    model3.to(device)
    optimizer3=torch.optim.SGD(list(model3.fc.parameters()) + list(model3.layer4.parameters()),lr=learningrate)
    train_val_test(model3, loss_fn, device, optimizer3, epoch, 'model3')