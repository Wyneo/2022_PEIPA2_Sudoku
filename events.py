

import pygame
from pygame.locals import *
from constantes import *
from drawGame import *
from utilsSudoku import *
import time

# fonction qui renvoit si elle existe la case de la grille sur laquelle l'utilisateur a cliqué
def clicOnGrid(mousepos, gridGUI):
    for case in gridGUI:
        rect = pygame.Rect(gridGUI[case]['Position'][0], gridGUI[case]['Position'][1], gridGUI[case]['Size'], gridGUI[case]['Size'])
        if rect.collidepoint(mousepos):
            return case
    return None

# fonction qui active (/change la couleur) d'une case
def highlight_case(coordinates, gridGUI):
    gridGUI[coordinates]["Color_Case"]=clairTRANSPARENCE


# fonction qui désactive (/change la couleur) d'une case
def unhighlight_case(coordinates, gridGUI):
    gridGUI[coordinates]["Color_Case"] = TRANSPARENT

def unhighlight_all(gridGUI):
    for i in range(9):
        for j in range(9):
            unhighlight_case((i,j), gridGUI)

# fonction qui initie la valeur d'une case
def inputNumber(coordinates, number, gridGUI, G, G_sol, mode):
    gridGUI[coordinates]['Value']=int(number)
    locked_case = updateColorCase(coordinates, gridGUI, G, G_sol, mode,number)
    return (locked_case)

# fonction qui initie la valeur d'une case à None
def eraseNumber(coordinates, gridGUI):
    gridGUI[coordinates]['Value']=None


# fonction qui ajoute une nouvelle valeur dans le jeu Sudoku
def addValue(G,coordinates,value) :
    G[coordinates[0]][coordinates[1]]=value

# fonction qui supprime une valeur dans le jeu Sudoku
def removeValue(G,coordinates) :
    G[coordinates[0]][coordinates[1]] = 0


# fonction qui met à jour la couleur
def updateColorCase(coordinates, gridGUI, G, G_sol, mode,number) :
    gridGUI[coordinates]['Color_Txt'] = BLACK
    #MODE DE JEU 1
    if (mode==1) :
        if number==G_sol[coordinates[0]][coordinates[1]]:
            gridGUI[coordinates]['Color_Txt'] = GREEN
    #MODE DE JEU 2
    if (mode==2) :
        if (not is_authorized(gridGUI[coordinates]['Value'], coordinates, G)):
            gridGUI[coordinates]['Color_Txt'] = RED
            return True
    return False


def FonctionsBoutons(i,mode,screen,current_highlighted,entree,G,G_sol,gridGUI):
    unhighlight_all(gridGUI)
    nvmode=mode
    BoutonEnfonce(i,screen)
    pygame.display.flip()
    aide=None
    if i == 0 :
        nvmode = 0
    elif i == 1:
        nvmode = 1
    elif i == 2:
        nvmode = 2
    elif i == 3:
        aide=Aide1(G, current_highlighted)
    elif i == 4:
        Aide2(G, entree, gridGUI,current_highlighted)
    elif i == 5:
        gridGUI = initGameGUI((50,50), 57, G_sol)
    return nvmode,aide,gridGUI


