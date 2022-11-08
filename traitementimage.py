import numpy as np
import cv2
from matplotlib import pyplot as plt

grille = cv2.imread('grille1.png')
grillegrise=cv2.cvtColor(grille, cv2.COLOR_RGB2GRAY)
#plt.figure()
#plt.imshow(grille, cmap=plt.cm.gray)
#plt.show()

def seuillage(grillegrise):
    img=grillegrise.copy()
    for i in range(grille.shape[0]):
        for j in range(grille.shape[1]):
            if grillegrise[i,j] < 105:
                img[i,j] = 0
            else:
                img[i,j] = 255
    #plt.imshow(img, cmap=plt.cm.gray)
    #plt.show()
    return img

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


img=seuillage(grillegrise)
contour=Contour(grille,img)
imax1=ContourMax(contour)
contour.pop(imax1)
imax2=ContourMax(contour)

contourGrille=contour[imax2]
grillecopy2 = grille.copy()
cv2.drawContours(grillecopy2,contourGrille,-1,(0,255,0),1)
#plt.imshow(grillecopy2)
#plt.show()

def SimplContour(contourGrille):
    peri=cv2.arcLength(contourGrille,True)
    simplc=cv2.approxPolyDP(contourGrille,0.1*peri,True)
    cv2.drawContours(grillecopy2,simplc,-1,(0,255,0),1)
    #plt.imshow(grillecopy2)
    #plt.show()
    return simplc

def orderContourPoints(contour):
    new_contour=[contour[0], contour[3],contour[2],contour[1]]
    return new_contour

def Redresser(contourGrille,grille):
    simplc=SimplContour(contourGrille)
    simplc=orderContourPoints(simplc)
    M1=np.zeros([900,900])
    pts1=np.float32(simplc)
    pts2=np.float32([[900,0],[0,0],[0,900],[900,900]])
    M2=cv2.getPerspectiveTransform(pts1,pts2)
    grillecopy3=grille.copy()
    rows,cols=M1.shape
    imgDroite=cv2.warpPerspective(grillecopy3,M2,(cols,rows))
    return imgDroite

imgDroite=Redresser(contourGrille,grille)
plt.imshow(imgDroite)
plt.show()



cases=[]
numzone=[]
coord=[]

grillecopy4=imgDroite.copy()

imgDG=cv2.cvtColor(imgDroite, cv2.COLOR_RGB2GRAY)
contour2=Contour(imgDroite,imgDG)

def SeparCase(grillecopy4,cases,coord):
    k=0
    for i in range(0,899,100):
        for j in range(0,899,100):
            cases.append(grillecopy4[i:i+100,j:j+100])
            #plt.imshow(cases[k])
            #plt.show()
            coord.append([[i,i+100],[j,j+100]])
            k+=1
    return cases

cases2=SeparCase(grillecopy4,cases,coord)

def RedimAndSave(img,x,y):
    imgfinale=cv2.resize(img,(x,y), cv2.INTER_CUBIC)
    #plt.imshow(imgfinale)
    #plt.show()
    plt.imsave('ImgFinale.jpeg',imgfinale)
    return imgfinale

imgfinale=RedimAndSave(grillecopy4,500,500)
