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


"""
Définition du réseau de neurones
Les réseaux sont définis par des classes. Vous pouvez définir plusieurs types de réseaux dans le même code et n'en instancier qu'un seul.
Ici la classe **Net** définit un réseau convolutif simple avec 5 couches de convolution et une couche de pooling.
"""

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

device = torch.device("cpu")
model = Net().to(device)
print(model)

"""
Fonction d'apprentissage
Il s'agit de la partie principale du programme. Elle utilise les données d'apprentissage et applique le gradient pour mettre à jour les paramètres du réseaux.
"""

def train( model, device, train_loader, optimizer,loss_function):
    train_loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #on remet à 0 les paramètres du gradient
        output = model(data) #on passe l'image dans le model (forward)
        loss = loss_function(output, target) #on calcul la function de coût (loss)
        train_loss += loss_function(output, target, reduction="sum").item() #on calcul la loss pour toute l'epoch

        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward() #retropropagation du gradient (backward)
        optimizer.step() #descente du gradient (optimize)

    train_loss /= len(train_loader.dataset)

    return train_loss, float(correct) / len(train_loader.dataset)


# Fonction de validation
# Il s'agit de la partie qui permet d'évaluer le modèle pendant l'entrainement avec des données qui non pas été vu par le réseau


def test(model, device, test_loader, loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target, reduction="sum").item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, float(correct) / len(test_loader.dataset)


# Chargement du modèle
path_modele = "./model/model.ckpt"
device = torch.device("cpu")
model = Net().to(device)
model.load_state_dict(torch.load(path_modele))
model.to(device)
model.eval()

#on applique la même transformation que les images de MNIST
transform=transforms.Compose([
    transforms.ToTensor(), # on met les images en tensor
    transforms.Normalize((0.1307,), (0.3081,)), # on normalise comme les images d'entrainement
    lambda x: x>0, #on binarise
    lambda x: x.float(),
])


def centrage(im):
    com = ndimage.measurements.center_of_mass(im)


    # Translation distances in x and y axis

    x_trans = int(im.shape[0]//2-com[0])
    y_trans = int(im.shape[1]//2-com[1])

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

from torch.utils.data import Dataset

import torch

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import glob
import random

class PrintedMNIST(Dataset):
    """Generates images containing a single digit from font"""

    def __init__(self, N, random_state, transform=None):
        """"""
        self.N = N
        self.random_state = random_state
        self.transform = transform

        fonts_folder = "fonts"

        # self.fonts = ["Helvetica-Bold-Font.ttf", 'arial-bold.ttf']
        self.fonts = glob.glob(fonts_folder + "/*.ttf")

        random.seed(random_state)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        img = Image.new("RGB", (28,28), (256,256,256)) #on créer l'image

        target = random.randint(1, 9)

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(random.choice(self.fonts), 25)
        draw.text((7, -3),str(target),(0,0,0),font=font)

        img = img.resize((28, 28), Image.BILINEAR)
        img = np.array(img)
        img = cv2.bitwise_not(img) #on inverse les intensités
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #on passe en niveau de gris

        if self.transform:
            img = self.transform(img)

        return img, target

train_set = PrintedMNIST(50000, -666, transform)
test_set = PrintedMNIST(5000, 33, transform)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=10)

dataiter = iter(train_loader)
images,labels = dataiter.next()
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i,0], cmap="gray") # only grayscale image here
plt.show()
print(labels)

# Chargement du modèle MNIST
path_modele = "./model/model.ckpt"
device = torch.device("cpu")
model = Net().to(device)
model.load_state_dict(torch.load(path_modele))

batch_size = 64
test_batch_size = 1000
lr = 0.01
momentum = 0.5
epochs = 1
loss_function = F.nll_loss
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=10, shuffle=True)


for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, loss_function)
    test_loss, test_acc = test(model, device, test_loader, loss_function)
    print('\nEpoch : {:.0f}, Train loss: {:.4f}, Validation loss: {:.4f}, Accuracy: {:.0f}/{} ({:.0f}%)\n'.format(
        epoch, train_loss, test_loss, test_acc*len(test_loader.dataset), len(test_loader.dataset),
        test_acc*100))

test_batch_size = 10
test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size, shuffle=True)
dataiter = iter(test_loader)
images,labels = dataiter.next()
model.eval()
output = model(images.to(device))
pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
pred = pred.view_as(labels)

nb_images = 30
prediction = []
correct = 0
target = [5,3,7,6,1,9,5,9,8,6,8,6,3,4,8,3,1,7,2,6,6,2,8,4,1,9,5,8,7,9]

for i in range (0,nb_images) :
    num = cv2.imread("./data_sudoku/num_"+str(i)+'.png')
    num_gray = cv2.cvtColor(num, cv2.COLOR_RGB2GRAY)
    num_resize = cv2.resize(num_gray, (28, 28))
    num_resize = clear_border(num_resize,1)
    num_resize = centrage(num_resize)
    num_tensor_transform = transform(num_resize)
    num_tensor_transform = torch.unsqueeze(num_tensor_transform, 0)
    model.float()
    output = model(num_tensor_transform.to(device))
    pred = output.argmax(dim=1, keepdim=True)
    prediction.append(pred.item())
    if pred == target[i] :
        correct += 1
    plt.imshow(num_tensor_transform[0,0], cmap="gray")
    #plt.show()


print(prediction)
print(correct/nb_images)
