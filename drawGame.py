# -*- coding: utf-8 -*-
#Importation des bibliothèques nécessaires
import pygame
from pygame.locals import *
from utilsGUI import *
from constantes import *

# fonction qui initialise la grille de l'interface du jeu
# Retourne une variable gridGUI = matrice de case à afficher
# Case = dictionnaire
# Paramètres d'entrée : P0 -> position en haut à gauche de la grille, square_size -> taille d'une case, G-> grille de sudooku = matrice
def initGameGUI(P0, square_size, G) :
    gridGUI = {}
    for i in range(9) :
        for j in range(9) :
            position = (P0[1]+j*square_size,P0[0]+i*square_size)
            if G[i][j] == 0 :
                case = {'Value':None, 'Position': position, 'Size':square_size, 'Color_Txt': BLACK, 'Color_Case': TRANSPARENT}
                gridGUI[(i,j)] = case
    return gridGUI

# fonction qui dessine le jeu
def drawGame(screen, gridGUI, grid_image, P0) :
    # Création d'une surface pour dessiner le jeu
    background = pygame.Surface(screen.get_size(), pygame.SRCALPHA, 32)
    background = background.convert_alpha()
    background.fill((250, 250, 250, 255))

    # Affichage du titre
    font = pygame.font.Font(None, 36)
    text = font.render("Sudoku", True, BLACK)
    textpos = text.get_rect()
    textpos.centerx = background.get_rect().centerx
    textpos.y = 10
    background.blit(text, textpos)

    #Ajouter l'image de la grille
    background.blit(grid_image, P0)
    drawGrid(background, gridGUI)

    #"Blitter" dans la fenêtre
    screen.blit(background,(0,0))


# fonction qui dessine la grille
def drawGrid(background, gridGUI):
    # pour chaque case de la grille, on dessine la case
    for case in gridGUI:
        drawCase(background,gridGUI[case])



# fonction qui dessine une case de la grille
def drawCase(background, case):
    # on dessine un rectangle de la taille de la case et de la couleur de celle-ci
    if case['Color_Case'] != None :
        Dcase = pygame.Surface((case["Size"]-4,case["Size"]-4), pygame.SRCALPHA, 32)
        Dcase = Dcase.convert_alpha()
        Dcase.fill(case['Color_Case'])

    # si la case a une valeur, on ajoute ce texte dans la case
    if (case['Value'] != None):
        font = pygame.font.Font(None, 40)
        text = font.render(str(case['Value']), True, case["Color_Txt"])
        textpos = text.get_rect()
        textpos.centerx = Dcase.get_rect().centerx
        textpos.centery = Dcase.get_rect().centery
        Dcase.blit(text, textpos)
    # on "blitte" la case dans le background
    background.blit(Dcase, (case["Position"][0], case["Position"][1]))

