import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

#on applique une transformation sur les images de MNIST afin de les normaliser
transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x>0, #binarisation des images
        lambda x: x.float(),
                       ])

batch_size = 10
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)

test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True)


# Visualisation des données
# La fonction iter() permet de créer un itérateur et la fonction next() permet d'obtenir le prochain batch.
# Vous pouvez exécuter plusieurs fois la cellule suivante pour voir différents jeux de données.

dataiter = iter(train_loader)
images,labels = dataiter.next()

for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(images[i, 0], cmap="gray") # only grayscale image here
print(labels)
plt.show()

# Définition du réseau de neurones
# Les réseaux sont définis par des classes. Vous pouvez définir plusieurs types de réseaux dans le même code et n'en instancier qu'un seul.
# Ici la classe **Net** définit un réseau convolutif simple avec 5 couches de convolution et une couche de pooling.

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

# Fonction d'apprentissage
# Il s'agit de la partie principale du programme. Elle utilise les données d'apprentissage et applique le gradient pour mettre à jour les paramètres du réseaux.

def train( model, device, train_loader, optimizer,loss_function):
    train_loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #on remet à 0 les paramètres du gradient
        output = model(data) #on passe l'image dans le model (forward)
        loss = loss_function(output, target) #on calcul la function de coût (loss)
        train_loss += loss_function(output, target, reduction='sum').item() #on calcul la loss pour toute l'epoch

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
            test_loss += loss_function(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, float(correct) / len(test_loader.dataset)


# Boucle d'apprentissage
# Fonction de coût et optimiseur
# Il faut d'abord définir la fonction de coût. Nous utiliserons la fonction déjà définie dans pytorch pour le négative log de vraissemblance, bien adaptée pour la tâche de classification.
# Ensuite il faut choisir un optimiseur qui explicite la façon d'appliquer le gradient calculé. Nous utiliserons la fonction classique Stochastic Gradient Descent utilisant un taux d'apprentissage (learning rate _lr_) et un momentum.
# Vous pouvez essayer de changer les paramètres comme le batch_size, la learning rate, le momentum, le nombre d'époque ou la function de coût (https://pytorch.org/docs/stable/nn.functional.html) pour voir si vous réussissez à améliorer les résultats d'apprentissage.

device = torch.device("cpu")
batch_size = 64
test_batch_size = 1000
lr = 0.01
momentum = 0.5
epochs = 5
loss_function = F.nll_loss

train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size, shuffle=True)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, loss_function)
    test_loss, test_acc = test(model, device, test_loader, loss_function)
    print('\nEpoch : {:.0f}, Train loss: {:.4f}, Validation loss: {:.4f}, Accuracy: {:.0f}/{} ({:.0f}%)\n'.format(
        epoch, train_loss, test_loss, test_acc*len(test_loader.dataset), len(test_loader.dataset),
        test_acc*100))


# Visualisation des résultats

test_batch_size = 10
test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size, shuffle=True)
dataiter = iter(test_loader)
images,labels = dataiter.next()
model.eval()
output = model(images.to(device))
pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
pred = pred.view_as(labels)



for i in range(test_batch_size):
        plt.subplot(1, test_batch_size, i+1)
        plt.imshow(images[i, 0], cmap="gray") # only grayscale image here
print(labels)
print(pred)
print(images.shape)
plt.show()


# Sauvegarde du modèle

path_modele = "./model/model.ckpt"
torch.save(model.state_dict(), path_modele)


# Chargement du modèle
model.load_state_dict(torch.load(path_modele))
model.to(device)
model.eval()

