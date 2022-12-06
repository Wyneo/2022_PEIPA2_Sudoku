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
                case = {'Value':None, 'Position': position, 'Size':square_size, 'Color_Txt': BLACK, 'Color_Case': TRANSPARENT,'Ver':0}
                gridGUI[(i,j)] = case
            else:
                case = {'Value':G[i][j], 'Position': position, 'Size':square_size, 'Color_Txt': BLACK, 'Color_Case': TRANSPARENT,'Ver':1}
                gridGUI[(i, j)] = case

    return gridGUI

# fonction qui dessine le jeu
def drawGame(screen, gridGUI, grid_image, P0,template,b,t) :
    # Création d'une surface pour dessiner le jeu
    background = pygame.Surface(screen.get_size(), pygame.SRCALPHA, 32)
    background = background.convert_alpha()
    background.fill(WHITE)

    #Ajouter le template de fond
    background.blit(template, (0,0))
    #Ajouter l'image de la grille
    background.blit(grid_image, P0)
    drawGrid(background, gridGUI)
    #Ajout des boutons
    PosBoutons=[(640,240),(770, 240),(900, 240),(1050, 240),(1180, 240)]

    B0=pygame.image.load("TemplateB/Boutons/B0.png")
    B1=pygame.image.load("TemplateB/Boutons/B1.png")
    B2=pygame.image.load("TemplateB/Boutons/B2.png")
    B3=pygame.image.load("TemplateB/Boutons/B3.png")
    B4=pygame.image.load("TemplateB/Boutons/B4.png")

    background.blit(B0, PosBoutons[0])
    background.blit(B1, PosBoutons[1])
    background.blit(B2, PosBoutons[2])
    background.blit(B3, PosBoutons[3])
    background.blit(B4, PosBoutons[4])

    if b != None :
        adresse = "TemplateB/Boutons/SB"+str(b)+".png"
        img = pygame.image.load(adresse)
        background.blit(img, PosBoutons[b])

    text= texteConsole(t)
    background.blit(text, (650, 390))
    pygame.display.flip()

    #"Blitter" dans la fenêtre
    screen.blit(background,(0,0))
    drawGrid(background, gridGUI)

# fonction qui dessine la grille
def drawGrid(background, gridGUI):
    # pour chaque case de la grille, on dessine la case
    for case in gridGUI:
        drawCase(background,gridGUI[case])



# fonction qui dessine une case de la grille
def drawCase(background, case):
    # on dessine un rectangle de la taille de la case et de la couleur de celle-ci
    if case['Color_Case'] != None :
        Dcase = pygame.Surface((case["Size"]-2,case["Size"]-2), pygame.SRCALPHA, 32)
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

def texteConsole(i):
    console = pygame.Surface((800,200), pygame.SRCALPHA, 32)
    console = console.convert_alpha()
    console.fill(TRANSPARENT)
    font = pygame.font.Font(None, 40)
    textpos = console.get_rect()
    if i==None:
        text = font.render("Amusez-vous bien !", True, BLACK)
    elif i==0:
        text = font.render("Ce mode de jeu n'apporte pas d'aide au joueur", True, BLACK)
    elif i==1:
        text = font.render("Ce mode de jeu affiche les entrées correctes en vert", True, BLACK)
    elif i==2:
        text = font.render("Ce mode de jeu affiche les entrées impossibles en rouge", True, BLACK)
    elif i==3:
        text = font.render("Cet indice donne le nombre de possibilités pour la case séléctionnée", True, BLACK)
    elif i==4:
        text = font.render("Cet indice donne les différentes cases pour le nombre entré", True, BLACK)
    elif i==V:
        text = font.render("Victoire !", True, BLACK)
    return text


def BoutonEnfonce(i,screen):
    PosBoutons = [(640, 240), (770, 240), (900, 240), (1050, 240), (1180, 240)]
    adresse = "TemplateB/Boutons/DB" + str(i) + ".png"
    img = pygame.image.load(adresse)
    screen.blit(img, PosBoutons[i])
