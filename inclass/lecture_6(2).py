import numpy as np, os, sys

import matplotlib.pyplot as plt #patch-wise similarities, droi images
from matplotlib import ticker, cm

import torch.nn as nn
import torch.utils.data 

import torch.optim

def datagen2d_m(mean2, flip, num):
  
  a1=np.pi*0.0
  mat1=np.array([ [np.cos(a1), np.sin(a1)  ], [ -np.sin(a1), np.cos(a1)] ] ) 

  a2=np.pi*0.2
  mat2=np.array([ [np.cos(a2), np.sin(a2)  ], [ -np.sin(a2), np.cos(a2)] ] ) 


  z1=np.random.normal(size=(num//2,2))
  z1[:,0]=z1[:,0]*3 
  x1=np.dot(z1,mat1) 
  z2=np.random.normal(size=(num-num//2,2))
  z2[:,0]=z2[:,0]*3 
  x2=np.dot(z2,mat2) + mean2.reshape((1,2))

  y1= (np.random.ranf(size=(num//2)) >= flip ).astype(dtype=np.float32)  #np.ones((num//2))+
  y2=  (np.random.ranf(size=(num-num//2)) <= flip ).astype(dtype=np.float32) #np.zeros((num-num//2,2))

  # random label noise in y1, y2

  x=np.concatenate((x1,x2),axis=0) #existing axis
  y=np.concatenate((y1,y2),axis=0) #existing axis

  #x.shape=(numdata,dims) dims=2 here
  #y.shape=(numdata)

  print('means',np.mean(y1),np.mean(y2),np.mean(y))
  print(x.shape,y.shape)

  # randomly permute
  inds=np.arange(num)
  np.random.shuffle(inds)
  x=x[inds,:]
  y=y[inds]

  return x,y

def rndsplit_simple(x,y,numtr):

  inds=np.arange(y.size)
  np.random.shuffle(inds)

  xtr=x[inds[0:numtr],:]
  ytr=y[inds[0:numtr]]

  xv=x[inds[numtr:],:]
  yv=y[inds[numtr:]]

  return xtr,ytr,xv,yv



def visualize_data(xv,yv,w,bias):
  
  print(yv)
  possamples=xv[ yv>0, : ]
  negsamples=xv[ yv<=0, : ]

  plt.plot(negsamples[:,0],negsamples[:,1],'bx')
  plt.plot(possamples[:,0],possamples[:,1],'rx')

  #plot wx+b=0 ... wx= -b, x= a w^O + w/\|w\|^2 * -b


  if 0==1:
    a=np.linspace(-10,10,200)
    worthogonal = np.asarray ( [ -w[1] ,  w[0]   ] ) 
    normedwtimesbias= -bias * w / np.linalg.norm(w)

    points=  a*  worthogonal / np.linalg.norm(w) + normedwtimesbias
    points=points.T
    print(points.shape, w.shape)  

    plt.plot(points[:,0],points[:,1],'c-', linewidth=5)

  if 2==2:
    delta=0.05
    x = np.arange(-10.0, 12.0, delta)
    y = np.arange(-10.0, 12.0, delta)
    X, Y = np.meshgrid(x, y)  

    U = bias + w[0]*X+ w[1]*Y  
    Z= 1.0/(1.0+np.exp(-U))

    CS = plt.contourf(X, Y, Z, levels=8,cmap=cm.viridis) #coolwarm

  plt.show()

def gendata():
  mean2=np.asarray([1,3])
  flip=0.1
  num=5000
  x,y=datagen2d_m(mean2, flip, num)
  numtr=x.shape[0]//2
  xtr,ytr,xv,yv=rndsplit_simple(x,y,numtr)

  return xtr,ytr,xv,yv

class logreglayer(nn.Module):
  def __init__(self,dims):
    
    super(logreglayer, self).__init__() #initialize base class

    self.bias=torch.nn.Parameter(data=torch.zeros(1).double(),requires_grad=True) #torch.nn.Parameter is a subclass of Tensor which is marked in a neural network module as parameter to be optimized, this is useful 1. for setting gradients to zero. 2. when using optimizer classes to tell them what parameters are optimized (next lecture)
    self.w=torch.nn.Parameter(data=torch.randn( (dims,1) ).double(),requires_grad=True) # random init shape must be (dims,1), requires_grad to True

  def forward(self,x):
    #TODO
    # YOUR IMPLEMENTATION HERE
    return 1/ (1+ torch.exp(-(torch.mm(x, self.w) + self.bias)))


def run():

  #torch.autograd lecture #pytorch custom layer has backward pass ?

  # gradient lecture: simple quadform 


  #training batch size
  batch_size=8
  #validation batch size
  valbatch_size=32

  # number of epochs
  maxnumepochs=12

  # learning rate
  learningrate=0.01

  
  #define dataset
  xtr,ytr,xv,yv=gendata()

  #Tensordataset ?
  #TODO
  dtr= torch.utils.data.TensorDataset(torch.tensor(xtr).double(), torch.tensor(ytr).double())#TensorDataset from tensors from xtr, ytr - our training features and labels
  dv= torch.utils.data.TensorDataset(torch.tensor(xv).double(), torch.tensor(yv).double()) # TensorDataset from tensors from xv, yv - our validation features and labels


  #define dataloader over dataset  
  loadertr=torch.utils.data.DataLoader(dtr,batch_size=batch_size,shuffle=True) # returns an iterator
  loaderval=torch.utils.data.DataLoader(dv,batch_size=valbatch_size,shuffle=False)

  # define your model class
  #TODO
  model= logreglayer(xtr.shape[1]) # your logreglayer properly initialized

  # optimizer, run it at first with the optimizer step, then replace it by your own version in step 2
  optimizer=torch.optim.SGD(model.parameters(),lr=learningrate, momentum=0.0, weight_decay=0)

  #define a loss from torch.nn.*
  #TODO
  lossfunction= nn.BCELoss() # which loss function suits here, given that our model produces 1-dimensional output  and we want to use it for classification?

  # define the device things will run on
  device=torch.device('cpu')

  bestclserror=-1

  for epoch in np.arange(maxnumepochs):

    print('at epoch', epoch)

    #TRAIN PHASE

    model.train(mode=True)
    avgloss=0
    for ct,data in enumerate(loadertr):


      model.zero_grad() # here it helps to have defined model.w and model.bias as torch.nn.Parameter - zero_grad acts on all parameters, this sets gradients to zero of all model parameters.

      #iterator from  torch.utils.data.TensorDataset yields a tuple ''data'', so data[0] returns a minibatch from xtr
      features=data[0]
      labels=data[1]

      # we dont move data yet to the right device
      features=features.to(device)
      labels=labels.to(device)

      #run prediction
      preds=model(features)

      # compute loss to ground truth
      loss= lossfunction(preds,labels)

      loss.backward() #computes gradient for every element in the minibatch

      #compute average loss for statistics
      avgloss+=loss.item()
      #print('loss',loss.item())


      #apply gradient to your parameters in model ... model.w and model.bias ... remember about data and grad :) 
      #TODO
      # run it at first using the optimizer, then replace it by your own version which updates the model parameters
      optimizer.step()
      # model.w += learningrate * model.w.grad.data
      # model.b += learningrate * model.b.grad.data

    print('epoch training loss',avgloss/(ct+1.0))

    #VAL PHASE

    model.train(mode=False)
    #number of samples encountered
    totalcount=0

    #number of classification errors encountered
    tmpclserror=0
    for ct,data in enumerate(loaderval):

      features=data[0]
      labels=data[1]

      # move data to the right device
      features=features.to(device)
      labels=labels.to(device)

      #get prediction on val
      with torch.no_grad():
        preds=model(features)
        totalcount+=labels.size()[0]
        # get classification error on val # works because outputs are probabilities, so threshold is 0.5
        tmpclserror+= torch.sum( (preds.squeeze(1)<0.5) & (labels==1) ) +  torch.sum( (preds.squeeze(1)>=0.5) & (labels==0) )

    #average the error by sample size
    clserror=tmpclserror.item()/float(totalcount)

    print('clserror',clserror)
    print('accuracy',1-clserror)
    # if val loss better than best so far ,save model
    if (bestclserror < 0) or (clserror < bestclserror):
      bestclserror=clserror
      bestweights=model.state_dict()
      print('found better model')
      #print(bestweights)

  #access the members of the model state dictionary
  w=bestweights['w'].numpy()
  bias=bestweights['bias'].item()

  # some eye candy
  visualize_data(xv,yv,w,bias)

  return bestweights




if __name__=='__main__':

  run()


