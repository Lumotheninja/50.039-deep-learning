import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt

class FlowersDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, dataset_file, transform):
        self.img_dir = img_dir
        self.dataset_file = dataset_file
        self.transform = transform

    def __len__(self):
        with open(self.dataset_file) as file:
            lines = file.readlines()
        return len(lines)
    
    def __getitem__(self, idx):
        with open(self.dataset_file) as file:
            img_name, label = file.readlines()[idx].strip('\n').split(' ')
            img_name = os.path.join(self.img_dir, img_name)
            img = self.transform(Image.open(img_name))
        return img, label

def train(model, device, train_loader, optimizer):
    model.train()
    samples_trained = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        bs, ncrops, c, h, w = data.size()
        optimizer.zero_grad()
        output = model(data.view(-1, c, h, w))
        output = output.view(bs, ncrops, -1).mean(1).argmax(dim=1)
        loss = F.nll_loss(output, target, reduction='sum')
        training_loss+= loss.item()
        loss.backward()
        optimizer.step()
        samples_trained += data.size()[0]
        print('Training. Gone through %d samples\r' %samples_trained, end="")
    training_loss /= samples_trained
    return training_loss


def test(model, device. test_loader):
    model.eval()
    correct_pred = 0
    samples_tested = 0
    test_loss = 0
    with torch.no_grad():
        for idx, (data,target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, c, h, w))
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            output = output.view(bs, ncrops, -1).mean(1).argmax(dim=1)
            correct_pred += torch.sum(output==target).item()
            samples_tested += data.size()[0]
            print('Testing. Gone through %d samples\r' %samples_tested end="")
    accuracy = correct_pred/samples_tested
    test_loss /= samples_tested
    return accuracy, test_loss
        
def no_preloaded_model(epoch):

    # define dataset and loaders
    train_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/trainfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))
    val_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/valfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))
    test_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/testfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=1)

    #define models and hyperparams
    model = models.resnet18(pretrained=False)
    device = torch.device('gpu')
    learningrate=0.01
    optimizer=torch.optim.SGD(model.parameters(),lr=learningrate, momentum=0.0, weight_decay=0)

    training_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    lowest_accuracy = 0
    highest_val_loss = 1

    # training and validating across epochs
    for t in range(epoch):
        training_loss = train(model, device, device, train_loader, optimizer)
        accuracy, val_loss = test(model, device, val_loader)
        if val_loss < highest_val_loss:
            highest_val_loss = val_loss
            torch.save(model.state_dict(), 'no_preloaded_model_best_weights.pt')
        val_accuracy_list.append(accuracy)
        training_loss_list.append(training_loss)
        val_lost_list.append(val_loss)

    # plot and save graph
    plt.plot(val_loss_list, label='val loss')
    plt.plot(training_loss_list, label='training loss')
    plt.plot(val_accuracy_list, label='val accuracy')
    plt.legend(loc='upper left')
    plt.savefig('no_preloaded_model.png')

    # testing time
    model.load_state_dict(torch.load('no_preloaded_model_best_weights.pt'))
    test_accuracy, _ = test(model, device, test_loader)
    with open('no_preloaded_model_best_weights.txt', 'w') as txtfile:
        file.write("Accuracy is: ", test_accuracy)

def preloaded_model(epoch):

    # define dataset and loaders
    train_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/trainfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))
    val_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/valfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))
    test_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/testfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=1)

    #define models and hyperparams
    model = models.resnet18(pretrained=True)
    device = torch.device('gpu')
    learningrate=0.01
    optimizer=torch.optim.SGD(model.parameters(),lr=learningrate, momentum=0.0, weight_decay=0)

    training_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    lowest_accuracy = 0
    highest_val_loss = 1

    # training and validating across epochs
    for t in range(epoch):
        training_loss = train(model, device, device, train_loader, optimizer)
        accuracy, val_loss = test(model, device, val_loader)
        if val_loss < highest_val_loss:
            highest_val_loss = val_loss
            torch.save(model.state_dict(), 'preloaded_model_best_weights.pt')
        val_accuracy_list.append(accuracy)
        training_loss_list.append(training_loss)
        val_lost_list.append(val_loss)

    # plot and save graph
    plt.plot(val_loss_list, label='val loss')
    plt.plot(training_loss_list, label='training loss')
    plt.plot(val_accuracy_list, label='val accuracy')
    plt.legend(loc='upper left')
    plt.savefig('preloaded_model.png')

    # testing time
    model.load_state_dict(torch.load('preloaded_model_best_weights.pt'))
    test_accuracy, _ = test(model, device, test_loader)
    with open('preloaded_model_best_weights.txt', 'w') as txtfile:
        file.write("Accuracy is: ", test_accuracy)

def last2_preloaded_model(epoch):
    # define dataset and loaders
    train_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/trainfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))
    val_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/valfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))
    test_dataset = FlowersDataset('flowersstuff/flowers_data/jpg', 
                           'flowersstuff/testfile.txt', 
                           transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=1)

    #define models and hyperparams
    model = models.resnet18(pretrained=True)
    device = torch.device('gpu')
    learningrate=0.01
    optimizer=torch.optim.SGD(list(resnet18.fc.params) + list(resnet18.layer4.params),lr=learningrate, momentum=0.0, weight_decay=0)

    training_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    lowest_accuracy = 0
    highest_val_loss = 1

    # training and validating across epochs
    for t in range(epoch):
        training_loss = train(model, device, device, train_loader, optimizer)
        accuracy, val_loss = test(model, device, val_loader)
        if val_loss < highest_val_loss:
            highest_val_loss = val_loss
            torch.save(model.state_dict(), 'last2_preloaded_model_best_weights.pt')
        val_accuracy_list.append(accuracy)
        training_loss_list.append(training_loss)
        val_lost_list.append(val_loss)

    # plot and save graph
    plt.plot(val_loss_list, label='val loss')
    plt.plot(training_loss_list, label='training loss')
    plt.plot(val_accuracy_list, label='val accuracy')
    plt.legend(loc='upper left')
    plt.savefig('last2_preloaded_model.png')

    # testing time
    model.load_state_dict(torch.load('last2_preloaded_model_best_weights.pt'))
    test_accuracy, _ = test(model, device, test_loader)
    with open('last2_preloaded_model_best_weights.txt', 'w') as txtfile:
        file.write("Accuracy is: ", test_accuracy)


if __name__ == '__main__':
    #no_preloaded_model(30)
    #preloaded_model(30)
    last2_preloaded_model(30)