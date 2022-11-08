

import pygame
from pygame.locals import *
from constantes import *
from drawGame import *
from utilsSudoku import *

# fonction qui renvoit si elle existe la case de la grille sur laquelle l'utilisateur a cliqué
def clicOnGrid(mousepos, gridGUI):
    for case in gridGUI:
        rect = pygame.Rect(gridGUI[case]['Position'][0], gridGUI[case]['Position'][1], gridGUI[case]['Size'], gridGUI[case]['Size'])
        if rect.collidepoint(mousepos):
            return case
    return None

# fonction qui active (/change la couleur) d'une case
def highlight_case(coordinates, gridGUI):
    gridGUI[coordinates]["Color_Case"]=BLUE


# fonction qui désactive (/change la couleur) d'une case
def unhighlight_case(coordinates, gridGUI):
    gridGUI[coordinates]["Color_Case"] = TRANSPARENT


# fonction qui initie la valeur d'une case
def inputNumber(coordinates, number, gridGUI, G, G_sol, mode):
    gridGUI[coordinates]['Value']=int(number)
    locked_case = updateColorCase(coordinates, gridGUI, G, G_sol, mode)
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
def updateColorCase(coordinates, gridGUI, G, G_sol, mode) :
    gridGUI[coordinates]['Color_Txt'] = BLACK
    #MODE DE JEU 1
    if (mode==1) :
        if G[coordinates[0]][coordinates[1]]==G_sol[coordinates[0]][coordinates[1]]:
            gridGUI[coordinates]['Color_Txt'] = GREEN
    #MODE DE JEU 2
    if (mode==2) :
        if (not is_authorized(gridGUI[coordinates]['Value'], coordinates, G)):
            gridGUI[coordinates]['Color_Txt'] = RED
            return True
    return False

