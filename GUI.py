# -*- coding: utf-8 -*-
#Importation des bibliothèques nécessaires
import pygame
from pygame.locals import *
from drawGame import *
from events import *
import threading
import time


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
    #-------------------------------------Initialisation pygame---------------------------------------------------------
    pygame.init()
    G = G_init.copy()
    window_H = 700                                                                                                      #Taille fenêtre de jeu
    window_W = 1500

    #--------------------------------Importation des images de fond-----------------------------------------------------
    grid_image=pygame.image.load("TemplateB/grilleVide2.png")
    template = pygame.image.load("TemplateB/TemplateTotale.png")

    #------------------------------Fenêtre de jeu + infos sur les cases-------------------------------------------------
    case_size = 57                                                                                                      #Taille d'une case
    P0 = (50,50)                                                                                                        #Marge écran-grille

    screen = pygame.display.set_mode((window_W, window_H))
    gridGUI = initGameGUI(P0, case_size, G)

    #--------------------------------Initialisation de variables nécéssaires--------------------------------------------
    b = None
    t = None
    tb = None
    mode = 0;                                                                                                           #Mode de jeu à 0 par défaut (mode normal)
    TempsStart = 86400 * time.localtime()[2] + 3600 * time.localtime()[3] + 60 * time.localtime()[4] + time.localtime()[5] #Heure en secondes du moment de lancement du sudoku
    current_highlighted = None                                                                                          #Représente, si elle existe, une case sélectionnée sur l'interface graphique
    locked_case = False                                                                                                 #Variable permettant de verrouiller une case
    entree = None
    continuer = 1                                                                                                       # Continuer ou interrompre la boucle

    #--------------------------------------------Dessin du jeu----------------------------------------------------------
    drawGame(screen, gridGUI, grid_image, P0, template, b, t, tb, TempsStart)
    pygame.display.flip()


    #----------------------------------------Infos sur les boutons------------------------------------------------------
    posBoutons=[(640,240),(770, 240),(900, 240),(1050, 240),(1180, 240),(1310, 240)]
    largeurBoutons = 117
    hauteurBoutons = 114

    #-------Lancement d'un thread en parallèle du main, rafraichissement de l'écran de jeu------------------------------
    thb = threading.Thread(target=RefreshTimer, args=(screen,TempsStart,))
    thb.start()

    #----------------------------------------Début de la boucle de jeu--------------------------------------------------
    while continuer:
        #----------------------------------------On reçoit un évènement-------------------------------------------------
        for event in pygame.event.get():
            #-------------------------------------Quitter la boucle-----------------------------------------------------
            if event.type == QUIT:
                continuer = 0

            # -------------------------------------Survol d'un bouton---------------------------------------------------
            if event.type == pygame.MOUSEMOTION:
                mousepos = pygame.mouse.get_pos()
                b=None
                for i in range(len(posBoutons)):
                    if posBoutons[i][0]<=mousepos[0]<=(posBoutons[i][0]+largeurBoutons) and posBoutons[i][1]<=mousepos[1]<=(posBoutons[i][1]+largeurBoutons):
                        b=i
                        drawGame(screen, gridGUI, grid_image, P0, template, b, t, tb, TempsStart)
                        pygame.display.flip()


            #-----------------------------------------Clic souris-------------------------------------------------------
            if event.type == pygame.MOUSEBUTTONDOWN:
                nbBouton = None
                mousepos = pygame.mouse.get_pos()                                                                       #Récupérer position souris
                #-----------------------------------Clic sur un bouton--------------------------------------------------
                for i in range(len(posBoutons)):
                    if posBoutons[i][0]<=mousepos[0]<=(posBoutons[i][0]+largeurBoutons) and posBoutons[i][1]<=mousepos[1]<=(posBoutons[i][1]+largeurBoutons) and not locked_case:
                        nbBouton=i
                        t=i
                        mode,tb,gridGUI = FonctionsBoutons(nbBouton,mode,screen,current_highlighted,entree,G,G_sol,gridGUI)
                        drawGame(screen, gridGUI, grid_image, P0, template, b, t, tb,TempsStart)
                        pygame.display.flip()
                #-----------------------------------Clic sur une case --------------------------------------------------
                if (not locked_case and 50<mousepos[0]<550 and 50<mousepos[1]<550 and nbBouton==None and gridGUI[clicOnGrid(mousepos, gridGUI)]['Ver']==0 ) :                                              #Case non verrouillée
                    if current_highlighted != None:                                 #Retirer surlignage
                        unhighlight_case(current_highlighted,gridGUI)
                    current_highlighted = clicOnGrid(mousepos, gridGUI)             #Nouveau surlignage
                    if current_highlighted != None:
                         highlight_case(current_highlighted, gridGUI)
                    drawGame(screen, gridGUI, grid_image, P0, template, b, t, tb, TempsStart)
                    pygame.display.flip()

            #----------------------------Entrée clavier + case sélectionnée---------------------------------------------
            if event.type == pygame.KEYDOWN and current_highlighted != None :

                #-------------------------Chiffres (pavé num & clavier)-------------------------------------------------
                key_1=1073741913
                key_9=1073741921
                cle_1=49
                cle_9=57
                entree = event.key
                if key_1<=entree<=key_9:
                    nombre = entree - 1073741912
                    locked_case = inputNumber(current_highlighted,nombre,gridGUI,G,G_sol,mode)  #Verrouillage de case + insertion nombre dans la grille
                    if not locked_case :
                        addValue(G, current_highlighted, nombre)
                if cle_1<=entree<=cle_9:
                    nombre = entree - 48
                    locked_case = inputNumber(current_highlighted, nombre, gridGUI, G, G_sol,  mode)  # Verrouillage de case + insertion nombre dans la grille
                    if not locked_case:
                        addValue(G, current_highlighted, nombre)

                #----------------------------------------Suppression----------------------------------------------------
                if entree==8:
                    eraseNumber(current_highlighted, gridGUI)
                    removeValue(G, current_highlighted)
                    locked_case = False

                #-----------------------------------Mise à jour de l'écran----------------------------------------------
                b = None
                if G == G_sol:
                    t="V"
                drawGame(screen, gridGUI, grid_image, P0,template,b,t,tb,TempsStart)
                pygame.display.flip()

#-----------------------------------------------------------------------------------------------------------------------
"""
Fonction RefreshTimer(screen,TempsStart): Réaffiche le timer toutes les [dt] secondes
Entrées : TempsStart (heure de début du sudoku en secondes), screen (écran de jeu)
Sortie : Affichage graphique
"""
def RefreshTimer(screen,TempsStart):
    while tha.is_alive():
        dt=0.05
        time.sleep(dt)
        Affichetimer(timer(TempsStart),screen)
        pygame.display.flip()

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    tha = threading.Thread(target=main)

    tha.start()
    tha.join()

