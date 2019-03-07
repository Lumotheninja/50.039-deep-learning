# Resnet18 with Imagenet dataset

Tested [Pytorch Resnet18](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) on the [Imagenet ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/index) challenge  
No training was involved, testing was done with 4 different transforms to see the power of data augmentation and adaptive pooling layers  

# Results

Test accuracy with 4 different transforms:  
1. Without normalization: 0.4456
2. With normalization: 0.6696 (This corresponded to top 1 error of [30.24%](https://pytorch.org/docs/stable/torchvision/models.html))
3. FiveCrop: 0.7292
4. Bigger resize: 0.6984
