import torchvision.models as models
import torchvision.transforms as transforms
import torch
import os
from collections import defaultdict
from PIL import Image
import numpy as np

is_dev = False


class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, data_type, transform):
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform
        with open(os.path.join(root_dir, "ImageSets", "Main", "%s.txt" % data_type)) as f:
            self.files = [line.strip() for line in f]

        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']
        self.labels = None
        if data_type != "test":
            labels = []
            for c in self.classes:
                with open(os.path.join(root_dir, "ImageSets", "Main", "%s_%s.txt" % (c, data_type))) as f:
                    labels.append([float(line.strip().split()[1]) for line in f])

            self.labels = (np.vstack(labels).T + 1)/2

    def __len__(self):
        return 50 if is_dev else len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img = Image.open(os.path.join(
            self.root_dir, "JPEGImages", img_name + '.jpg'))
        img = self.transform(img)
        return (img, self.labels[idx]) if self.labels is not None else img


def AP(output, target):
    ap = 0
    for i in range(output.shape[1]):
        o = output[:,i]
        t = target[:,i]
        order = o.argsort()[::-1]
        ap += np.mean(np.array([t[order][:j+1].mean()
                                for j in range(o.shape[0])])[t[order] == 1])
    return ap/output.shape[1]


def evaluate(dataloader, model, device, lossfunction=None):
    model.eval()
    ap = 0
    total = 0
    count = 0
    loss = 0
    predictions = []
    #targets = []
    for ct, data in enumerate(dataloader):
        count += 1
        print("Evaluating...%08d" % total, end="\r")
        features = data.to(device)
        #labels = data[1].to(device).squeeze()

        total += features.shape[0]

        preds = model(features)

        if lossfunction:
            loss += lossfunction(preds, labels).item()

        predictions.append(preds.cpu().detach().numpy())
        #targets.append(labels.cpu().detach().numpy())

        del features
        #del labels
        del preds

    #mAP = AP(np.concatenate(predictions), np.concatenate(targets))
    #return predictions, (mAP, loss/count) if lossfunction else mAP
    return predictions, None


def train_model(train, val, model, optimizer, device, lossfunction, name="model", maxnumepochs=50):
    best_map = 0

    train_losses = []
    val_losses = []
    val_maps = []

    for epoch in np.arange(maxnumepochs):

        print('at epoch', epoch)

        # TRAIN PHASE

        model.train(mode=True)
        avgloss = 0
        seen = 0
        count = 0

        for ct, data in enumerate(train):
            count += 1

            model.zero_grad()

            features = data[0]
            labels = data[1].squeeze()

            seen += labels.shape[0]

            # we dont move data yet to the right device
            features = features.to(device)
            labels = labels.to(device)

            # run prediction
            preds = model(features)

            # compute loss to ground truth
            loss = lossfunction(preds, labels)

            loss.backward()  # computes gradient for every element in the minibatch

            # compute average loss for statistics
            avgloss += loss.item()

            optimizer.step()

            del features
            del labels

            print('epoch training loss (%5d) %8f' %
                  (seen, avgloss/(ct+1)), end='\r')
        print('epoch training loss (%5d) %8f' %
              (seen, avgloss/count))
        train_losses.append(avgloss/count)

        # val
        val_map, val_loss = evaluate(val, model, device, lossfunction)
        val_losses.append(val_loss)
        val_maps.append(val_map)
        print("epoch val loss %8f" % val_loss)
        print("epoch val map %8f" % val_map)

        # if val loss better than best so far ,save model
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), "%s.pt" % name)
            print('found better model')

        print("====================================================")
    # save figures
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.title("train loss vs epoch")
    plt.savefig("%s_train_loss.png" % name)
    plt.close()
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.title("val loss vs epoch")
    plt.savefig("%s_val_loss.png" % name)
    plt.close()
    plt.plot(np.arange(len(val_maps)), val_maps)
    plt.title("val map vs epoch")
    plt.savefig("%s_val_map.png" % name)
    plt.close()


def multi_label_loss(output, target):
    loss = 0
    for i in range(target.shape[1]):
        loss += torch.nn.BCEWithLogitsLoss()(
            output[:, i].double(), target[:, i].double())
    return loss/target.shape[1]


test_dataset = VOCDataset('./VOCdevkit/VOC2012/', 'test', transforms.Compose([transforms.Resize(
    size=280), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

batch_size = 8 if is_dev else 32
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

device = torch.device('cpu')
# define hyperparams
model = models.resnet34(pretrained=True)
model.fc = torch.nn.Linear(512, 20)
model.load_state_dict(torch.load("model6.pt"))
model.to(device)

predictions, mAp= evaluate(test_dataloader, model, device)
print(mAp)

p = np.concatenate(predictions)
np.save("predictions.npy", p)
for i, c in enumerate(test_dataset.classes):
    with open("results/VOC2012/Main/comp1_cls_test_%s.txt" %c, "w") as f:
        for name, value in zip(test_dataset.files, p[:, i]):
            f.write("%s %f\n"%(name, value))



