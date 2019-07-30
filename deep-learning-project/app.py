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
            image = "./static/VOCdevkit/VOC2012/JPEGImages/" + request.form['url'] + ".jpg"
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
        num_lines = 0
        top_images = []
        for line in f.readlines():
            num_lines += 1
            top_images.append(line.strip('\n').split(' '))
        top_images.sort(key=lambda tup: float(tup[1]))
        top_images.reverse()
    num_pages = num_lines//20 + 1
    page = request.args.get('page', 1, type=int)
    display_images = top_images[(page-1)*20:] if (page == num_pages) else top_images[(page-1)*20:page*20] 
    return render_template('top_image.html', class_name=class_name, top_images=display_images, num_pages=num_pages, page=page)

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
    output = output.view(bs, ncrops, -1).mean(1)
    sigmoid = torch.sigmoid(output)
    plt.bar(classes, sigmoid.squeeze().detach().numpy(), label='sigmoid value')
    plt.xticks(rotation=90)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('./static/sigmoid.png')
    plt.clf()
    preds = sigmoid > 0.5
    labels = list(np.where(pred==1)[0] for pred in preds)[0]
    return ([classes[label] for label in labels])

if __name__ == "__main__":    
    app.run(host='0.0.0.0')
