# -*- coding: utf-8 -*-
#Importation des bibliothèques nécessaires
import pygame
from pygame.locals import *
from drawGame import *
from Recochiffre import *
from SolveurSat import *
from events import *
from Images import *
import threading
import time
import os

##------------------------Début des fonctions--------------------------------
"""
Fonction main(): Fontion principale du programme
Entrées : -
Sortie : -
"""
def main():
    #-------------------------------------Initialisation pygame---------------------------------------------------------
    os.environ['SDL_VIDEO_CENTERED'] = '1'  # You have to call this before pygame.init()

    pygame.init()

    info = pygame.display.Info()


    pygame.init()
    G = G_init.copy()
    window_W = info.current_w                                                                                                  #Taille fenêtre de jeu
    window_H = info.current_h
    window_W, window_H = window_W, window_H - 60
    WC = window_W / 1500
    #------------------------------Fenêtre de jeu + infos sur les cases-------------------------------------------------
    case_size = 57 *WC                                                                                                     #Taille d'une case
    P0 = (50*WC,50*WC)                                                                                                        #Marge écran-grille

    screen = pygame.display.set_mode((window_W, window_H))
    pygame.display.update()

    gridGUI = initGameGUI(P0, case_size, G,WC)

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
    drawGame(screen, gridGUI, grid_image, P0, template, b, t, tb, TempsStart,WC)
    pygame.display.flip()


    #----------------------------------------Infos sur les boutons------------------------------------------------------
    posBoutons=[(640*WC,240*WC),(770*WC, 240*WC),(900*WC, 240*WC),(1050*WC, 240*WC),(1180*WC, 240*WC),(1310*WC, 240*WC)]
    largeurBoutons = 117*WC

    #-------Lancement d'un thread en parallèle du main, rafraichissement de l'écran de jeu------------------------------
    thb = threading.Thread(target=RefreshTimer, args=(screen,TempsStart,WC,))
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
                        drawGame(screen, gridGUI, grid_image, P0, template, b, t, tb, TempsStart,WC)
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
                        mode,tb,gridGUI = FonctionsBoutons(nbBouton,mode,screen,current_highlighted,entree,G,G_sol,gridGUI,WC)
                        drawGame(screen, gridGUI, grid_image, P0, template, b, t, tb,TempsStart,WC)
                        pygame.display.flip()
                #-----------------------------------Clic sur une case --------------------------------------------------
                if (not locked_case and 50*WC<mousepos[0]<550*WC and 50*WC<mousepos[1]<550*WC and nbBouton==None and gridGUI[clicOnGrid(mousepos, gridGUI)]['Ver']==0 ) :                                              #Case non verrouillée
                    if current_highlighted != None:                                 #Retirer surlignage
                        unhighlight_case(current_highlighted,gridGUI)
                    current_highlighted = clicOnGrid(mousepos, gridGUI)             #Nouveau surlignage
                    if current_highlighted != None:
                         highlight_case(current_highlighted, gridGUI)
                    drawGame(screen, gridGUI, grid_image, P0, template, b, t, tb, TempsStart,WC)
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
                drawGame(screen, gridGUI, grid_image, P0,template,b,t,tb,TempsStart,WC)
                pygame.display.flip()

#-----------------------------------------------------------------------------------------------------------------------
"""
Fonction RefreshTimer(screen,TempsStart): Réaffiche le timer toutes les [dt] secondes
Entrées : TempsStart (heure de début du sudoku en secondes), screen (écran de jeu)
Sortie : Affichage graphique
"""
def RefreshTimer(screen,TempsStart,WC):
    continuer = True
    while tha.is_alive() and continuer:
        dt=0.05
        time.sleep(dt)
        Affichetimer(timer(TempsStart),screen,WC)
        pygame.display.flip()

#-------------Initialisation des valeurs des grilles (à adapter)-------------

G_init = fonction()

if G_init != None and (SolveurSat(G_init) != None) :
    for i in range(len(G_init)):
        print(G_init[i])
    G = G_init.copy()

    G_sol = SolveurSat(G_init)

    if __name__ == "__main__":
        tha = threading.Thread(target=main)

        tha.start()
        tha.join()
else:
    img = cv2.imread ("TemplateB/messageerreur/erreursolution.png")
    cv2.namedWindow('erreur', cv2.WINDOW_NORMAL)
    cv2.imshow('erreur', img)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()




