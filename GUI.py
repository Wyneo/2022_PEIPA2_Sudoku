# -*- coding: utf-8 -*-
#Importation des bibliothèques nécessaires
import pygame
from pygame.locals import *
from drawGame import *
from events import *
import threading


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
"""
G_init = [[0,0,0, 0,0,0, 0,0,0],
          [0,0,0, 0,0,0, 0,0,0],
          [0,0,0, 0,0,0, 0,0,0],
          [0,0,0, 0,0,0, 0,0,0],
          [0,1,0, 0,0,0, 0,0,0],
          [0,0,0, 0,0,0, 0,0,0],
          [0,0,0, 0,0,0, 0,0,0],
          [0,0,0, 0,0,0, 0,0,0],
          [0,0,0, 0,0,0, 0,0,0]]
"""
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
    G_temp = G_init.copy()
    window_H = 700
    window_W = 1500
    #Importer image
    grid_image=pygame.image.load("TemplateB/grilleVide2.png")
    case_size = 57
    P0 = (50,50)       #Marge écran-grille
    #Importer le fond
    template = pygame.image.load("TemplateB/TemplateTotale.png")
    #Séléction du mode de jeu (0->sans aide, 1->couleur verte pour chiffres corrects, 2->couleur rouge pour chiffres interdits à un emplacement
    mode = 0;

    #Création fenêtre de jeu et dessin du jeu
    screen = pygame.display.set_mode((window_W, window_H))
    thb = threading.Thread(target=timer,args=(screen,))
    thb.start()
    gridGUI = initGameGUI(P0, case_size, G_temp)
    b = None
    t = None
    drawGame(screen, gridGUI,grid_image, P0,template,b,t)
    pygame.display.flip()

    #À propos des boutons
    posBoutons = [(640, 240), (770, 240), (900, 240), (1050, 240), (1180, 240)]
    largeurBoutons = 117
    hauteurBoutons = 114

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

            if event.type == pygame.MOUSEMOTION:
                mousepos = pygame.mouse.get_pos()
                b=None
                for i in range(len(posBoutons)):
                    if posBoutons[i][0]<=mousepos[0]<=(posBoutons[i][0]+largeurBoutons) and posBoutons[i][1]<=mousepos[1]<=(posBoutons[i][1]+largeurBoutons):
                        b=i
                        drawGame(screen, gridGUI, grid_image, P0, template, b,t)
                        pygame.display.flip()


            #CLIC SOURIS
            if event.type == pygame.MOUSEBUTTONDOWN:
                nbBouton = None
                mousepos = pygame.mouse.get_pos()                                   #Récupérer position souris
                for i in range(len(posBoutons)):
                    if posBoutons[i][0]<=mousepos[0]<=(posBoutons[i][0]+largeurBoutons) and posBoutons[i][1]<=mousepos[1]<=(posBoutons[i][1]+largeurBoutons):
                        nbBouton=i
                        t=i
                        mode = FonctionsBoutons(nbBouton,mode,screen)
                        drawGame(screen, gridGUI, grid_image, P0, template, b, t)
                        pygame.display.flip()
                if (not locked_case and 50<mousepos[0]<550 and 50<mousepos[1]<550 and nbBouton==None and gridGUI[clicOnGrid(mousepos, gridGUI)]['Ver']==0 ) :                                              #Case non verrouillée
                    if current_highlighted != None:                                 #Retirer surlignage
                        unhighlight_case(current_highlighted,gridGUI)
                    current_highlighted = clicOnGrid(mousepos, gridGUI)             #Nouveau surlignage
                    if current_highlighted != None:
                         highlight_case(current_highlighted, gridGUI)
                    drawGame(screen, gridGUI,grid_image, P0,template,b,t)
                    pygame.display.flip()


            #ENTREE CLAVIER + CASE SÉLECTIONNÉE
            if event.type == pygame.KEYDOWN and current_highlighted != None :
                #CHIFFRES
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

                #SUPPRESSION
                if entree==8:
                    eraseNumber(current_highlighted, gridGUI)
                    removeValue(G, current_highlighted)
                    locked_case = False
                #MISE À JOUR ÉCRAN DE JEU
                b = None
                drawGame(screen, gridGUI, grid_image, P0,template,b,t)
                pygame.display.flip()



def timer(screen):
    num_of_secs=0

    console = pygame.Surface((800, 200), pygame.SRCALPHA, 32)
    console = console.convert_alpha()
    console.fill(RED)
    font = pygame.font.Font(None, 40)
    text = font.render("Ce mode de jeu n'apporte pas d'aide au joueur", True, BLACK)
    console.blit(text, (0, 0))
    screen.blit(console, (0, 0))
    pygame.display.flip()

    while tha.is_alive():
        m,s = divmod(num_of_secs, 60)
        min_sec_format = '{:02d}:{:02d}'.format(m, s)
        print(min_sec_format, end='\n')
        time.sleep(1)
        num_of_secs += 1



if __name__ == "__main__":
    tha = threading.Thread(target=main)

    tha.start()


    tha.join()

