import numpy as np
import cv2
from matplotlib import pyplot as plt

grille = cv2.imread('grille1.png')

'''
Fonction Seuillage : prend en compte une photo de grille en couleur, la transforme en nuance de gris.
Chaque pixel devient noir ou blanc selon sa nuance
Entrée : grille (image originale de la grille)
Sortie : img (image de la grille en noir et blanc)'''
def seuillage(grille):
    grillegrise = cv2.cvtColor(grille, cv2.COLOR_RGB2GRAY)
    th = grillegrise.copy()
    img = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 17)
    #plt.imshow(img)
    #plt.show()
    return img

"""
Contour(grille,img) : Récupère tous les contours présents sur l'image
Entrées : grille (image originale de la grille), img (grille en noir et blanc)
Sortie : contour (matrice contenant les contours)
"""
def Contour(grille,img):
    kernel = np.ones((3,3),np.uint8)
    grillecopy=grille.copy()
    #grillegrise=cv2.cvtColor(grille, cv2.COLOR_RGB2GRAY)
    _,thresh=cv2.threshold(img, 110, 255, 0)
    ouverture= cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel)
    contour,_=cv2.findContours(ouverture, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(grillecopy,contour,-1,(0,255,0),1)
    #plt.imshow(grillecopy)
    #plt.show()
    contour = list(contour)
    return contour

"""
ContourMax(contour) : Retourne la plus grande aire présente sur la photo
Entrée : contour (matrice contenant les contours)
Sortie : imax (indice du contour ayant la plus grande aire dans le tableau contour)
"""
def ContourMax(contour):
    aire = []
    for c in contour:
        aire.append(cv2.contourArea(c))
    max = 0
    imax = 0
    for i in range(len(aire)-1):
        if aire[i]>=max:
            imax=i
            max=aire[i]
    return imax

img = seuillage(grille)
contour = Contour(grille,img)
imax1 = ContourMax(contour)
contour.pop(imax1)
imax2 = ContourMax(contour)

contourGrille = contour[imax2]
grillecopy2 = grille.copy()
cv2.drawContours(grillecopy2,contourGrille,-1,(0,255,0),1)
plt.imshow(grillecopy2)
plt.show()

'''
Fonction VerifContour : vérifie que le contour selectionné est bien le contour de la grille
Entrée : contourGrille (le contour selectionné)
Sortie : booléen
'''
def VerifContour(contourGrille):
    "effectuer verif"
    if "condition verif":
        return True
    else : return False

"""
SimplContour(contourGrille): Trouve les coordonnées des points correspondant aux 4 coins de la grille
Entrée : contourGrille (grand contour de la grille -> le carré)
Sortie : simplc (tableau avec les coordonnées des points correspondant aux 4 coins de la grille)
"""
def SimplContour(contourGrille):
    #if VerifContour :
    peri=cv2.arcLength(contourGrille,True)
    simplc=cv2.approxPolyDP(contourGrille,0.1*peri,True)
    cv2.drawContours(grillecopy2,simplc,-1,(0,255,0),1)
    #plt.imshow(grillecopy2)
    #plt.show()
    return simplc
    'else : return False'

"""
orderContourPoints(contour): Crée un contour à partir des coordonnées des 4 coins de la grille
Entrée : contour (grand contour de la grille -> le carré)
Sortie : new_contour (contour créé à partir des coordonnées des 4 coins de la grille)
"""
def orderContourPoints(contour):
    new_contour=[contour[0],contour[3],contour[2],contour[1]]
    return new_contour

"""
Redresser(contourGrille,grille) : Application des deux fonctions précédentes afin de récupérer une image droite
Entrée : contourGrille (grand contour de la grille -> le carré), grille
Sortie : imgDroite (grilledroite, sans les éléments autour)
"""
def Redresser(contourGrille,grille):
    simplc=SimplContour(contourGrille)
    #if simplc:
    simplc=orderContourPoints(simplc)
    M1=np.zeros([900,900])
    pts1=np.float32(simplc)
    pts2=np.float32([[900,0],[0,0],[0,900],[900,900]])
    M2=cv2.getPerspectiveTransform(pts1,pts2)
    grillecopy3=grille.copy()
    rows,cols=M1.shape
    imgDroite=cv2.warpPerspective(grillecopy3,M2,(cols,rows))
    return imgDroite
    '''else :
        print("Image peu claire")
        return False'''

imgDroite=Redresser(contourGrille,grille)
<<<<<<< HEAD
#if imgDroite:
=======
>>>>>>> b15f2d69fc10cfd8f6060a6216dbe810bc3dfdc3
plt.imshow(imgDroite)
plt.show()

'''
Fonction SeparCase : Renvoie un tableau dont chaque élément est une case de la grille
Entrée : imgDroite (grilledroite, sans les éléments autour)
Sorties : cases : tableau comprenant chaque case de la grille, coord : tableau de coordonnées de chaque case
'''
def SeparCase(imgDroite):
    grillecopy4 = imgDroite.copy()
    k=0
    cases = []
    coord = []
    for i in range(0,899,100):
        for j in range(0,899,100):
            cases.append(grillecopy4[i:i+100,j:j+100])
            #plt.imshow(cases[k])
            #plt.show()
            coord.append([[i,i+100],[j,j+100]])
            k+=1
    return cases

<<<<<<< HEAD
#cases = SeparCase(grillecopy4,cases,coord)

'''
Fonction RedimAndSave : Redimentionne l'image traitée pour utilisation dans le projet et l'enrgistre en tant que fichier jpeg
Entrée : img (l'image à redmiensionner), x et y (dimensions de l'image finale souhaitée)
Sortie : imgfinale (l'image redimensionnée)
'''
def RedimAndSave(img,x,y):
    imgfinale=cv2.resize(img,(x,y), cv2.INTER_CUBIC)
    #plt.imshow(imgfinale)
    #plt.show()
    plt.imsave('ImgFinale.jpeg',imgfinale)
    return imgfinale
=======
cases2=SeparCase(grillecopy4,cases,coord)
>>>>>>> b15f2d69fc10cfd8f6060a6216dbe810bc3dfdc3

def RedimAndSave(img,x,y):
    imgfinale=cv2.resize(img,(x,y), cv2.INTER_CUBIC)
    #plt.imshow(imgfinale)
    #plt.show()
    plt.imsave('ImgFinale.jpeg',imgfinale)
    return imgfinale

imgfinale=RedimAndSave(grillecopy4,500,500)
