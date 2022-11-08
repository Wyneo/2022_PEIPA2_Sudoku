# -*- coding: utf-8 -*-
#Importation des bibliothèques nécessaires
import pygame
from pygame.locals import *
from drawGame import *
from events import *


#-------------Initialisation des valeurs des grilles (à adapter)-------------
G_init = [[5,0,0, 0,2,0, 8,0,9],
          [0,4,1, 8,0,0, 0,6,0],
          [0,0,2, 6,0,9, 3,0,0],
          [0,0,7, 5,0,8, 0,1,0],
          [0,9,0, 0,4,0, 5,0,7],
          [4,5,0, 0,0,1, 0,2,0],
          [6,0,4, 0,1,0, 0,0,2],
          [0,1,0, 7,0,0, 0,5,4],
          [8,0,0, 0,6,2, 1,0,0]]


G_sol = [[5,6,3, 1,2,4, 8,7,9],
          [9,4,1, 8,3,7, 2,6,5],
          [7,8,2, 6,5,9, 3,4,1],
          [3,2,7, 5,9,8, 4,1,6],
          [1,9,8, 2,4,6, 5,3,7],
          [4,5,6, 3,7,1, 9,2,8],
          [6,3,4, 9,1,5, 7,8,2],
          [2,1,9, 7,8,3, 6,5,4],
          [8,7,5, 4,6,2, 1,9,3]]

G=G_init.copy()

##------------------------Début des fonctions--------------------------------
"""
Fonction main(): Fontion principale du programme
Entrées : -
Sortie : -
"""
def main():
    #Initialisation Pygame
    pygame.init()
    G_temp = G_init
    window_H = 600
    window_W = 600
    #Importer image
    grid_image = pygame.image.load("Grille1_taille1.png")
    case_size = int(grid_image.get_height()//9);
    P0 = (50,50)

    #Séléction du mode de jeu (0->sans aide, 1->couleur verte pour chiffres corrects, 2->couleur rouge pour chiffres interdits à un emplacement
    mode = 2;

    #Création fenêtre de jeu et dessin du jeu
    screen = pygame.display.set_mode((window_H, window_W))
    gridGUI = initGameGUI(P0, case_size, G_temp)
    drawGame(screen, gridGUI,grid_image, P0)
    pygame.display.flip()


    continuer = 1                                                                   #Continuer ou interrompre la boucle

    #BOUCLE
    current_highlighted = None                                                      #Représente, si elle existe, une case sélectionnée sur l'interface graphique
    locked_case = False                                                             #Variable permettabt de verrouiller une case


    while continuer:
        #Évènements recus de l'utilisateur
        for event in pygame.event.get():
            #QUITTER LA BOUCLE
            if event.type == QUIT:
                continuer = 0
            
            #CLIC SOURIS
            if event.type == pygame.MOUSEBUTTONDOWN:
                mousepos = pygame.mouse.get_pos()                                   #Récupérer position souris
                if (not locked_case) :                                              #Case non verrouillée
                    if current_highlighted != None:                                 #Retirer surlignage
                        unhighlight_case(current_highlighted,gridGUI)
                    current_highlighted = clicOnGrid(mousepos, gridGUI)             #Nouveau surlignage
                    if current_highlighted != None:
                         highlight_case(current_highlighted, gridGUI)
                    drawGame(screen, gridGUI,grid_image, P0)
                    pygame.display.flip()

            #ENTREE CLAVIER + CASE SÉLECTIONNÉE
            if event.type == pygame.KEYDOWN and current_highlighted != None :
                #CHIFFRES
                key_1=1073741913
                key_9=1073741921
                entree = event.key
                if key_1<=entree<=key_9:
                    nombre = entree - 1073741912
                    mode = 2
                    locked_case = inputNumber(current_highlighted,nombre,gridGUI,G,G_sol,mode)  #Verrouillage de case + insertion nombre dans la grille
                    if not locked_case :
                        addValue(G, current_highlighted, nombre)
                #SUPPRESSION
                if entree==8:
                    eraseNumber(current_highlighted, gridGUI)
                    removeValue(G, current_highlighted)
                    locked_case = False
                #MISE À JOUR ÉCRAN DE JEU
                drawGame(screen, gridGUI, grid_image, P0)
                pygame.display.flip()







if __name__ == "__main__":
    main()
