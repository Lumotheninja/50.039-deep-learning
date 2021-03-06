import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from getimagenetclasses import *
from PIL import Image

class ImagenetTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, text_dir, img_dir, synset, transform):
        self.text_dir = text_dir
        self.img_dir = img_dir
        self.synset = synset
        self.transform = transform

    def __len__(self):
        return 2500
        
    def __getitem__(self, idx):
        img_file_name = 'ILSVRC2012_val_' + str(idx+1).zfill(8) + '.JPEG'
        txt_file_name = 'ILSVRC2012_val_' + str(idx+1).zfill(8) + '.xml'
        img_name = os.path.join(self.img_dir, img_file_name)
        txt_name = os.path.join(self.text_dir, txt_file_name)
        img = Image.open(img_name)
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        image = self.transform(rgbimg)
        indicestosynsets,synsetstoindices,synsetstoclassdescr = parsesynsetwords(self.synset)
        label,_ = parseclasslabel(txt_name,synsetstoindices)
        sample = (image, label)
        return sample

resnet18 = models.resnet18(pretrained=True)

def evaluate(model, dataloader, device, limit=2500):
    model.eval()
    correct_pred = 0
    images = 0
    with torch.no_grad():
        for data,target in dataloader:
            data, target = data.to(device), target.to(device)
            if data.size()[1] == 5:
                bs, ncrops, c, h, w = data.size()
                output = model(data.view(-1, c, h, w))
                output = output.view(bs, ncrops, -1).mean(1)
            else:
                output = model(data)
            pred = output.argmax(dim=1)
            correct_pred += torch.sum(pred==target).item()
            images += data.size()[0]
            print('Gone through %d images\r' %images, end="")
            if images == limit:
                break
    print ("")
    return correct_pred/images

def without_norm():
    print ("Testing without norm")
    dataset = ImagenetTrainingDataset(text_dir = 'data/ILSVRC2012_bbox_val_v3/val',
                                 img_dir = 'data/imagenet2500/imagespart',
                                 synset = 'data/synset_words.txt',
                                 transform = transforms.Compose([transforms.Resize(size=224), transforms.CenterCrop(224), transforms.ToTensor()]))
    ImagenetLoader = torch.utils.data.DataLoader(dataset,batch_size=25, num_workers=1)
    device = torch.device('cpu')
    model=resnet18
    print ("Accuracy: ", evaluate(model, ImagenetLoader, device))

def with_norm():
    print ("Testing with norm")
    dataset = ImagenetTrainingDataset(text_dir = 'data/ILSVRC2012_bbox_val_v3/val',
                                 img_dir = 'data/imagenet2500/imagespart',
                                 synset = 'data/synset_words.txt',
                                 transform = transforms.Compose([transforms.Resize(size=224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    ImagenetLoader = torch.utils.data.DataLoader(dataset,batch_size=25, num_workers=1)
    device = torch.device('cpu')
    model=resnet18
    print ("Accuracy: ", evaluate(model, ImagenetLoader, device))

def five_crop():
    print ("Testing Fivecrop")
    dataset = ImagenetTrainingDataset(text_dir = 'data/ILSVRC2012_bbox_val_v3/val',
                                 img_dir = 'data/imagenet2500/imagespart',
                                 synset = 'data/synset_words.txt',
                                 transform = transforms.Compose([transforms.Resize(size=280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))]))
    ImagenetLoader = torch.utils.data.DataLoader(dataset,batch_size=25, num_workers=1)
    device = torch.device('cpu')
    model=resnet18
    print ("Accuracy: ", evaluate(model, ImagenetLoader, device))

def bigger_size():
    print ("Testing bigger size")
    dataset = ImagenetTrainingDataset(text_dir = 'data/ILSVRC2012_bbox_val_v3/val',
                                 img_dir = 'data/imagenet2500/imagespart',
                                 synset = 'data/synset_words.txt',
                                 transform = transforms.Compose([transforms.Resize(size=330), transforms.CenterCrop(330), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    ImagenetLoader = torch.utils.data.DataLoader(dataset,batch_size=25, num_workers=1)
    device = torch.device('cpu')
    model=resnet18
    print ("Accuracy: ", evaluate(model, ImagenetLoader, device))

if __name__ == '__main__':
    without_norm()
    with_norm()
    five_crop()
    bigger_size()

