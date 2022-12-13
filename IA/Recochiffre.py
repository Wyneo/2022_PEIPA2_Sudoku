import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.segmentation import clear_border
from scipy import ndimage
from traitementimage import *


class GlobalAveragePooling(nn.Module):
    def __init__(self, in_channels):
        self.s = in_channels
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], 1, x.shape[2] * x.shape[3])
        x = x.mean(dim = 3, keepdim = True)
        return x

    def __repr__(self):
        return 'GlobalAveragePooling({}, {})'.format(self.s, self.s)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(30);
        self.conv2 = nn.Conv2d(30, 30, 3, stride=1)
        self.bn2 = nn.BatchNorm2d(30);
        self.conv3 = nn.Conv2d(30, 30, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(30);
        self.gap = GlobalAveragePooling(30)
        self.conv4 = nn.Conv2d(30, 30, 1, stride=1)
        self.bn4 = nn.BatchNorm2d(30);
        self.conv5 = nn.Conv2d(30, 10, 1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.gap(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def centrage(im):
    com = ndimage.center_of_mass(im)
    # Translation distances in x and y axis

    x_trans = int(im.shape[0]/2-com[0])
    y_trans = int(im.shape[1]/2-com[1])

    # Pad and remove pixels from image to perform translation

    if x_trans > 0:
        im2 = np.pad(im, ((x_trans, 0), (0, 0)), mode='constant')
        im2 = im2[:im.shape[0]-x_trans, :]
    else:
        im2 = np.pad(im, ((0, -x_trans), (0, 0)), mode='constant')
        im2 = im2[-x_trans:, :]

    if y_trans > 0:
        im3 = np.pad(im2, ((0, 0), (y_trans, 0)), mode='constant')
        im3 = im3[:, :im.shape[0]-y_trans]

    else:
        im3 = np.pad(im2, ((0, 0), (0, -y_trans)), mode='constant')
        im3 = im3[:, -y_trans:]

    return im3

transform=transforms.Compose([
    transforms.ToTensor(), # on met les images en tensor
    transforms.Normalize((0.1307,), (0.3081,)), # on normalise comme les images d'entrainement
    lambda x: x>0, #on binarise
    lambda x: x.float(),
])

#Chargement du model entrainé 2 fois
path_modele = "./model/model_bis.ckpt"
device = torch.device("cpu")
model = Net().to(device)
model.load_state_dict(torch.load(path_modele))
model.to(device)
model.eval()

nb_images = 81
prediction = []
correct = 0
resultat = []

""" 
Pour les 81 images du sudoku, récupéré grâce à traitementimage, 
on fait un seuillage, on inverse les couleurs pour correspondre aux images d'entrainement, 
on fait un clear_border, et si l'image n'est pas toute noir, on centre et resize pour correspondre aux images d'entrainements
puis le model nous renvoie le chiffre dans la case du tableau résultat,
sinon le programme met un zéro dans la case.
"""

for i in range (0,nb_images):
    num = cases[i]
    num_seuil = seuillage(num)
    num_seuil=255-num_seuil
    num_resize = clear_border(num_seuil,6)
    #plt.imshow(num_resize, cmap=plt.cm.gray)
    #plt.show()
    print(np.count_nonzero(num_resize))
    if np.count_nonzero(num_resize)>500:
        num_resize = centrage(num_resize)
        num_resize = cv2.resize(num_resize, (28, 28))
        num_tensor_transform = transform(num_resize)
        num_tensor_transform = torch.unsqueeze(num_tensor_transform, 0)
        #plt.imshow(num_tensor_transform[0, 0])
        #plt.show()
        model.float()
        output = model(num_tensor_transform.to(device))
        pred = output.argmax(dim=1, keepdim=True)
        resultat.append(pred.item())
    else :
        resultat.append(0)

"mise en forme des résultats sous la forme du tableau [[]*9] voulu pour la suite"
T=[[0,]*9,]*9
t=[]
k=0
for i in range(0,9):
    for j in range(0,9):
        t.append(resultat[k])
        T[i] = t
        k=k+1
    t=[]

print(T)
