import os,sys,numpy as np

import torch

import time

def forloopdists(feats,protos):
  dist = np.empty((feats.shape[0], protos.shape[0]))
  for i in range(feats.shape[0]):
    for j in range(protos.shape[0]):
      sum_feats = 0
      sum_protos = 0
      for k in range(feats.shape[1]):
        sum_feats += feats[i][k]
        sum_protos += protos[j][k]
      diff = sum_feats - sum_protos
      norm = np.power(diff, 2)
      dist[i][j] = norm
  return np.exp(- dist)
  #YOUR implementation here

def numpydists(feats,protos):
  featsnew = np.expand_dims(np.sum(feats, 1), 1)
  protosnew = np.expand_dims(np.sum(protos, 1), 0)
  diff = featsnew-protosnew
  norm = np.power(diff,2)
  return np.exp(- norm)
  #YOUR implementation here
  
def pytorchdists(feats0,protos0,device):
  featsnew = torch.unsqueeze(torch.einsum('ij->i',torch.tensor(feats0)), dim =1)
  protosnew = torch.unsqueeze(torch.einsum('ij->i',torch.tensor(protos0)), dim=0)
  diff = featsnew-protosnew
  norm = np.power(diff,2)
  return np.exp(- norm)
  #YOUR implementation here


def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##f
  ###############
  feats=np.random.normal(size=(10,300)) #5000 instead of 250k for forloopdists
  protos=np.random.normal(size=(50,300))


  
  since = time.time()
  dists0=forloopdists(feats,protos)
  time_elapsed=float(time.time()) - float(since)
  print('For loop complete in {:.3f}s'.format( time_elapsed ))
  print (dists0)

  device=torch.device('cpu')
  since = time.time()

  dists1=pytorchdists(feats,protos,device)


  time_elapsed=float(time.time()) - float(since)

  print('Pytorch complete in {:.3f}s'.format( time_elapsed ))
  print(dists1)

  #print('df0',np.max(np.abs(dists1-dists0)))


  since = time.time()

  dists2=numpydists(feats,protos) 


  time_elapsed=float(time.time()) - float(since)

  print('Numpy complete in {:.3f}s'.format( time_elapsed ))

  print(dists2)

  print('df',(np.abs(dists1-torch.tensor(dists2))).max())


if __name__=='__main__':
  run()
