import os
import gc
import torch
import argparse
import librosa
import matplotlib
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster.hierarchy as shc

from models import *
from dataset import *
from utils import progress_bar
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('==> Creating networks..')
rowcnn = RowCNN().to(device)
rowcnn.load_state_dict(torch.load('./weights/network_train.ckpt'))

idx_to_label = {
        0: 'Cello', # Cello
        1: 'Clarinet', # Clarinet
        2: 'Flute', # Flute
        3: 'Acoustic_guitar', # Acoustic guitar
        4: 'Electric_guitar', # Electric guitar
        5: 'Organ', # Organ
        6: 'Piano', # Piano
        7: 'Saxophone', # Saxophone
        8: 'Trumpet', # Trumpet
        9: 'Violin'  # Violin
        }

print('==> Loading data..')
testset = InstrumentsDataset(test=True)

l_list = list(rowcnn.children())
l_list = list(l_list[0].children())
features = torch.nn.Sequential(*l_list[:-1]) 

print(features)

def forward_loss(features, x):
    xs = []
    for conv in features:
        x2 = F.relu(conv(x))  
        x2 = torch.squeeze(x2, -1)      
        x2 = F.max_pool1d(x2, x2.size(2)) 
        xs.append(x2)
    x = torch.cat(xs, 2) 
    return x

for param in features.parameters():
    param.requires_grad = False

data_labels = []

def test_instruments():
    dataloader = DataLoader(testset, batch_size=1, shuffle=True)
    dataloader = iter(dataloader)

    data_array = []
    target_done = []

    for batch_idx in range(len(dataloader)):
        if(len(target_done) == 10):
            break
        #print(batch_idx)
        inputs, targets = next(dataloader)
        inputs = torch.tensor(inputs).type(torch.FloatTensor)
        inputs = inputs.to(device)

        feature_acts = forward_loss(features, inputs)
        feature_acts = feature_acts[0].detach().cpu().numpy()

        if(int(targets[0]) not in target_done):
            data_array.append(feature_acts.flatten())
            data_labels.append(idx_to_label[int(targets[0])])
            print(int(targets[0]))
            target_done.append(int(targets[0]))

        del inputs
        del targets
        gc.collect()
        torch.cuda.empty_cache()

    return data_array

def plot_clustering(X, labels, title=None):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./plots/cluster")

def llf(id):
    return data_labels[id]

def plot_dendogram(X, linkage) :
    plt.figure(figsize=(10, 7))  
    plt.title("Instruments Dendogram")  
    dend = shc.dendrogram(shc.linkage(X, method=linkage), leaf_label_func=llf, leaf_rotation=27)  
    plt.savefig("./plots/dendogram_{}".format(linkage))

def clusterformation(X) :
    for linkage in ('ward', 'average', 'complete', 'single'):
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
        t0 = time()
        clustering.fit(X)
        #plot_clustering(X, clustering.labels_, "%s linkage" % linkage)
        plot_dendogram(X, linkage)


if __name__ == '__main__':
    print("Testing Started")
    data_array = test_instruments()
    print("Clustering Started")
    #kmeans = KMeans(n_clusters=12, random_state=0).fit(data_array)
    #print(kmeans.labels_)
    for i in range(len(data_labels)):
        print("{} : {}".format(i, data_labels[i]))
    clusterformation(data_array)
   
