import pygame
from pygame.locals import *
from constantes import *
from events import *

#INITIALISATION

S=[[5,0,0,0,2,0,8,0,9],
   [0,4,1,8,0,0,0,6,0],
   [0,0,2,6,0,9,3,0,0],
   [0,0,7,5,0,8,0,1,0],
   [0,9,0,0,4,0,5,0,7],
   [4,5,0,0,0,1,0,2,0],
   [6,0,4,0,1,0,0,0,2],
   [0,1,0,7,0,0,0,5,4],
   [8,0,0,0,6,2,1,0,0]]
#--------------------------------------------------
"""
Fonction creation() : Créer le tableau P initial, avec toutes les possibilités pour un sudoku vide (ensembles de 1 à 9)
Entrées : -
Sorties : P (Liste de listes d'ensembles)
"""
def creation():
    P = []
    for i in range(9):
        L=[]
        for j in range(9):
            L.append({1,2,3,4,5,6,7,8,9})
        #print (L)
        P.append(L)
    return(P)

"""
Fonction remplacezero(S,P): Remplace dans P les cases ayant des valeurs définies
Entrées: S (matrice avec le sudoku), P (tableau avec toutes les possibilités - initial)
Sorties: P (tableau de possibilités modifié)
"""
def remplacezero(S,P):
    for i in range(9):
        for j in range(9):
            if S[i][j]!=0:
                P[i][j]=S[i][j]
    return P

"""
Fonction recupbloc(P,l,c) :  A partir d'un élément, la fonction récupère un bloc
Entrées : grille de base (P), indice ligne du premier élement du bloc (l), indice colonne du premier élement du bloc (c)
Sortie : une liste bloc
"""
def recupbloc(P,l,c):
    bloc=[]
    for i in range(l,l+3):
        for j in range(c,c+3):
            bloc.append(P[i][j])
    return(bloc)

"""
Fonction remplacebloc(P,L,l,c) : Remplace un bloc de P par les valeurs de L
Entrées: listes P,L, indice de ligne du premier élement du bloc (l), indice colonne du premier élement du bloc (c)
Sorties : - [P est modifié]
"""
def remplacebloc(P,L,l,c):
    a=0
    for i in range(3):
        for j in range(3):
            P[l+i][c+j]=L[a]
            a=a+1

"""
Fonction recupcolonne(S,i) : A partir d'un indice, la fonction récupère la colonne de l(indice indiqué
Entrées : grille de base (P), indice de la colonne (i)
 Sortie : une liste colonne
 """
def recupcolonne(P,i):
    colonne = []
    for j in range(0,9):
        colonne.append(P[j][i])
    return colonne

"""
Fonction remplacebloc(P,L,l,c) : Remplace un bloc de P par les valeurs de L
Entrées: listes P,L, indice colonne (j)
Sorties : - [P est modifié]
"""
def remplacecolonne(P,L,j):
    for i in range(9):
        P[i][j]=L[i]

"""
Fonction verif(L) : A partir d'une liste préalablement récupérée, 
    la fonction enlève les valeurs déjà présentes dans la liste des ensembles de possibilités des autres cases
Entrée : liste (L)
Sortie : liste (L) modifiée
"""
def verif(L):
    l=set()
    for i in L:
        if type(i)==int:
            l.add(i)
    for k in range (len(L)):
        if type(L[k])==set:
            L[k]=L[k]-l
    return(L)

"""
Fonction parcours(P)
Entrée: tableau valeur + possibilités (P)
Sortie : tableau valeur + possibilités simplifié (P)
"""
def parcours(P):
    for i in range(9):
        verif(P[i])
        remplacecolonne(P,verif(recupcolonne(P,i)),i)
    for j in range(0,9,3):
        for k in range(0,9,3):
            remplacebloc(P,verif(recupbloc(P,j,k)),j,k)
    return P

#--------------------------------------------------
"""
Fonction proche(P,l,c) : 
Entrée: tableau valeurs + possibilités (P), indice ligne de l'élement (l), indice colonne de l'élement (c)
Sortie : tableau valeurs + possibilités simplifié (P)
"""
def proche(P,l,c):
    verif(P[l])
    for x in range(9):
        estSolution(P,l,x)
    remplacecolonne(P,verif(recupcolonne(P,c)),c)
    for y in range(9):
        estSolution(P,y,c)
    if l<=2:
        la = 0
    elif l<=5:
        la= 3
    else :
        la = 6
    if c<=2:
        ca = 0
    elif c<=5:
        ca= 3
    else :
        ca = 6
    remplacebloc(P,verif(recupbloc(P,la,ca)),la,ca)
    for j in range(3):
        for k in range(3):
            estSolution(P,la+j,ca+k)
    return P

"""
Fonction estSolution(P) : Un ensemble à un unique élément devient l'entier correspondant
Entrées: tableau valeurs + possibilités (P), indice ligne de l'élement (l), indice colonne de l'élement (c)
Sorties: False ou - [P est modifié]
"""
def estSolution(P,l,c):
    if not(type(P[l][c])==set and len(P[l][c])==1):
        return False
    else:
        P[l][c]=list(P[l][c])[0]
        proche(P, l, c)

#--------------------------------------------------

"""
Fonction Aide1(G_temp,case): Lorsqu'une case est sélectionnée avec l'entrée clavier "i", le nombre de valeurs possibles pour cette case est affichée
Entrées: grille de sudoku temporaire + case sélectionnée
Sorties: Texte indiquant le nombre de valeurs possibles pour la case
"""
def Aide1(G_temp,case):
    P = remplacezero(G_temp, creation())
    proche(P,case[0],case[1])
    nb=0
    if (type(P[case[0]][case[1]])==set):
        nb= len(P[case[0]][case[1]])
        return "Il y a ",nb," valeur(s) possible(s)."
    if(type(P[case[0]][case[1]])==int):
        return "Il n'y a qu'une valeur possible."

"""
Fonction Solution1(G_temp,case): Lorsqu'une case est sélectionnée avec l'entrée clavier "s", s'il n'y a qu'une valeur possible pour cette case, cette valeur est affichée
Entrées: grille de sudoku temporaire + case sélectionnée
Sorties: Texte indiquant la solution de la case s'il n'y a qu'une valeur possible
"""

def Solution1(G_temp,case):
    P = remplacezero(G_temp, creation())
    proche(P, case[0], case[1])
    if (type(P[case[0]][case[1]]) == int):
        return "La valeur possible est ",P[case[0]][case[1]], "."

"""
Fonction Aide2(G_temp,x): Lorsqu'une entrée clavier chiffre (1,2,3,4,5,6,7,8,9) est effectuée, les endroits possibles pour ce chiffre sont print et affiché en bleue sur la grille 
Entrées: grille de sudoku temporaire + chiffre clavier + grille + taille d'une case
Sorties: Texte indiquant les cases possibles et les affichant en bleue 
"""

def Aide2(G_temp,x,gridGUI,square_size):
    P = remplacezero(G_temp, creation())
    for l in range(9):
        for c in range(9):
            proche(P,l,c)
            if (type(P[l][c]) == set):
                if (x-48) in P[l][c]:
                    position = (50 + c* square_size, 50 + l* square_size)
                    case = {'Value': None, 'Position': position, 'Size': square_size, 'Color_Txt': BLACK,
                            'Color_Case': BLUE2}  #Ajouter BLUE2 = (0,0, 255, 100) dans constantes
                    gridGUI[(l,c)]=case
                    print((x-48), "est dans la case de colonne ",c," et de ligne ",l)
                else:
                    position = (50 + c* square_size, 50 + l* square_size)
                    case = {'Value': None, 'Position': position, 'Size': square_size, 'Color_Txt': BLACK,
                            'Color_Case': TRANSPARENT}
                    gridGUI[(l,c)]=case
            if (type(P[l][c]) == int):
                if (x - 48) == P[l][c]:
                    position = (50 + c * square_size, 50 + l * square_size)
                    case = {'Value': None, 'Position': position, 'Size': square_size, 'Color_Txt': BLACK,
                            'Color_Case': BLUE2}
                    gridGUI[(l, c)] = case
                else:
                    position = (50 + c * square_size, 50 + l * square_size)
                    case = {'Value': None, 'Position': position, 'Size': square_size, 'Color_Txt': BLACK,
                            'Color_Case': TRANSPARENT}
                    gridGUI[(l, c)] = case