from flask import Flask, render_template, request
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']
model = models.resnet34()
model.fc = torch.nn.Linear(512, 20)
model.load_state_dict(torch.load("model6.pt", map_location='cpu'))
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html', classes = classes)

@app.route('/predict', methods=['POST'])
def predict_image_label():
    graph_url = None
    if request.method == 'POST':
        if "url" in request.form:
            image = "./static/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/" + request.form['url'] + ".jpg"
        else:
            image = request.files['image']
        try: 
            label = eval_one_image(image, classes, model)
            message = 'The image is classified as ' + ', '.join(label)
            graph_url = '/static/sigmoid.png?' + str(time.time())
        except Exception as e:
            message = str(e)
        return render_template('index.html', classes = classes, message = message, graph_url = graph_url)
    else:
        return render_template('index.html', classes = classes)

@app.route('/top_images/<class_name>', methods=['GET'])
def view_top_images(class_name):
    with open('results/comp1_cls_val_' + class_name + '.txt', 'r') as f:
        top_images = [line.strip('\n').split(' ') for line in f.readlines()]
        top_images.sort(key=lambda tup: float(tup[1]))
    return render_template('top_image.html', class_name=class_name, top_images=top_images[-30:][::-1])

def eval_one_image(image, classes, model):
    model.eval()
    img = Image.open(image)
    transform = transforms.Compose([transforms.Resize(size=280),
                                transforms.FiveCrop(224),
                                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]
                                                                            + [transforms.ToTensor()(transforms.functional.adjust_brightness(crop, 1.5)) for crop in crops])),
                                transforms.Lambda(lambda crops:
                                                  torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])
    data = transform(img).unsqueeze(dim=0)
    bs, ncrops, c, h, w = data.size()
    output = model(data.view(-1, c, h, w))
    output = output.view(bs, ncrops, -1).max(1)[0]
    sigmoid = torch.sigmoid(output)
    plt.bar(classes, sigmoid.squeeze().detach().numpy(), label='training loss')
    plt.xticks(rotation=90)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./static/sigmoid.png')
    plt.clf()
    preds = sigmoid > 0.5
    labels = list(np.where(pred==1)[0] for pred in preds)[0]
    return ([classes[label] for label in labels])