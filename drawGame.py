# -*- coding: utf-8 -*-
#Importation des bibliothèques nécessaires
import pygame
from pygame.locals import *
from utilsGUI import *
from constantes import *
from Aides import *
from Images import *
import time

# fonction qui initialise la grille de l'interface du jeu
# Retourne une variable gridGUI = matrice de case à afficher
# Case = dictionnaire
# Paramètres d'entrée : P0 -> position en haut à gauche de la grille, square_size -> taille d'une case, G-> grille de sudooku = matrice
def initGameGUI(P0, square_size, G,WC) :
    gridGUI = {}
    for i in range(9) :
        for j in range(9) :
            position = (P0[1] + j * square_size, P0[0] + i * square_size)
            numcase = (position[0] // square_size, position[1] // square_size)
            position = (position[0] - (0.7*WC*numcase[0]), position[1] - (0.7*WC*numcase[1]))
            if G[i][j] == 0 :
                case = {'Value':None, 'Position': position, 'Size':square_size, 'Color_Txt': BLACK, 'Color_Case': TRANSPARENT,'Ver':0}
                gridGUI[(i,j)] = case
            else:
                case = {'Value':G[i][j], 'Position': position, 'Size':square_size, 'Color_Txt': BLACK, 'Color_Case': TRANSPARENT,'Ver':1}
                gridGUI[(i, j)] = case

    return gridGUI



#-----------------------------------------------------------------------------------------------------------------------
"""
Fonction drawGame(screen, gridGUI, grid_image, P0,template,b,t,TempsStart): Dessiner le jeu
Entrées :   screen (écran de jeu), 
            gridGUI (dictionnaire, contient l'ensemble des informations sur les cases à dessiner), 
            grid_image (image de la grille), 
            P0 (marge entre la grille et le bord),
            template (image du fond),
            b (numéro du bouton survolé),
            t (numéro du texte à afficher dans la console),
            tb (numéro du texte à afficher dans la console, ligne 2),
            TempsStart (Heure de départ du jeu, en s)
Sortie : Affichage graphique
"""
def drawGame(screen, gridGUI, grid_image, P0,template,b,t, tb,TempsStart,WC) :
    #----------------------Création d'une surface pour dessiner le jeu--------------------------------------------------
    background = pygame.Surface(screen.get_size(), pygame.SRCALPHA, 32)
    background = background.convert_alpha()
    background.fill(FOND)
    #------------------------------Ajouter le template de fond----------------------------------------------------------
    background.blit(template, (0,0))

    #--------------------Ajouter l'image de la grille + dessin de la grille---------------------------------------------
    background.blit(grid_image, P0)
    drawGrid(background, gridGUI)
    #-----------------------------------Ajout des boutons---------------------------------------------------------------
    PosBoutons=[(640*WC,240*WC),(770*WC, 240*WC),(900*WC, 240*WC),(1050*WC, 240*WC),(1180*WC, 240*WC),(1310*WC, 240*WC)]

    background.blit(B0, PosBoutons[0])
    background.blit(B1, PosBoutons[1])
    background.blit(B2, PosBoutons[2])
    background.blit(B3, PosBoutons[3])
    background.blit(B4, PosBoutons[4])
    background.blit(B5, PosBoutons[5])

    if b != None :                                                                                                      #Si un bouton est survolé, on importe l'image du bouton survolé
        adresse = "TemplateB/Boutons/SB"+str(b)+".png"
        img = pygame.image.load(adresse)
        img = pygame.transform.scale(img, (117 * WC, 114 * WC))
        background.blit(img, PosBoutons[b])

    #--------------------------------Affichage texte dans la console----------------------------------------------------

    textA,textB= texteConsole(t,tb,WC)
    background.blit(textA, (650*WC, 390*WC))
    if textB!=None:
        background.blit(textB, (650*WC, 420*WC))

    #"Blitter" dans la fenêtre
    screen.blit(background,(0,0))
    Affichetimer(timer(TempsStart), screen,WC)
    drawGrid(background, gridGUI)



#------------------------------------Dessin grille + cases--------------------------------------------------------------
"""
Fonction drawGrid(background, gridGUI): Dessiner la grille
Entrées : background, gridGUI (dictionnaire, contient l'ensemble des informations sur les cases à dessiner)
Sortie : Affichage graphique
"""
def drawGrid(background, gridGUI):
    # pour chaque case de la grille, on dessine la case
    for case in gridGUI:
        drawCase(background,gridGUI[case])

"""
Fonction drawCase(background, case): Dessiner une case de la grille
Entrées : background, case (contient les informations sur la case à dessiner)
Sortie : Affichage graphique
"""
def drawCase(background, case):
    # on dessine un rectangle de la taille de la case et de la couleur de celle-ci
    if case['Color_Case'] != None :
        Dcase = pygame.Surface((case["Size"],case["Size"]), pygame.SRCALPHA, 32)
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

#-----------------------------------------------------------------------------------------------------------------------
"""
Fonction texteConsole(i): Écrire une phrase sur la première ligne de la console, dépendant du bouton enfoncé
Entrées : i (numéro du texte à aff)
Sortie : text (texte à écrire)
"""
def texteConsole(i,j,WC):
    font = pygame.font.Font(None, int(30*WC))
    textB = None
    if i==None:
        textA = font.render("Amusez-vous bien !", True, BLACK)
    elif i==0:
        textA = font.render("Ce mode de jeu n'apporte pas d'aide au joueur.", True, BLACK)
    elif i==1:
        textA = font.render("Ce mode de jeu affiche les entrées correctes en vert.", True, BLACK)
    elif i==2:
        textA = font.render("Ce mode de jeu affiche les entrées impossibles en rouge.", True, BLACK)
    elif i==3:
        textA = font.render("Cet indice donne le nombre de possibilités pour la case sélectionnée.", True, BLACK)
        if j != None :
            textB = font.render(j, True, BLACK)
    elif i==4:
        textA = font.render("Cet indice donne les différentes cases pour le nombre entré dans une case.", True, BLACK)
    elif i==5:
        textA = font.render("Voici la solution.", True, BLACK)
    elif i=="V":
        textA = font.render("Vous avez gagné !", True, BLACK)
        textB = font.render("Félicitations !", True, BLACK)
    return textA,textB


#--------------------------------Affichage des boutons cliqués----------------------------------------------------------
"""
Fonction BoutonEnfonce(i,screen): On importe la bonne image et on l'affiche à la position correcte
Entrées : i (numéro du bouton), screen (écran de jeu)
Sortie : Affichage graphique
"""
def BoutonEnfonce(i,screen,WC):
    PosBoutons = [(640*WC,240*WC),(770*WC, 240*WC),(900*WC, 240*WC),(1050*WC, 240*WC),(1180*WC, 240*WC),(1310*WC, 240*WC)]
    adresse = "TemplateB/Boutons/DB" + str(i) + ".png"
    img = pygame.image.load(adresse)
    img = pygame.transform.scale(img, (117 * WC, 114 * WC))
    screen.blit(img, PosBoutons[i])



#------------------------------------------Gestion du timer-------------------------------------------------------------
"""
Fonction timer(TempsStart) : Calcul un timer à partir de l'heure en seconde et de l'heure au début du sudoku
Entrées: TempsStart (Heure de départ du jeu, en s)
Sorties: min_sec_format (Timer au format mm:ss)
"""
def timer(TempsStart):
    TempsS = 86400 * time.localtime()[2] + 3600 * time.localtime()[3] + 60 * time.localtime()[4] + time.localtime()[5]
    timer = TempsS - TempsStart
    m, s = divmod(timer, 60)
    min_sec_format = '{:02d}:{:02d}'.format(m, s)
    return min_sec_format


"""
Fonction Affichertimer(m,screen): affiche le timer grâce à un temps donné
Entrées: m (temps à afficher), screen (écran de jeu)
Sorties: affichage graphique
"""
def Affichetimer(m,screen,WC):
    fond = pygame.Surface((85*WC, 30*WC), pygame.SRCALPHA, 32)
    fond = fond.convert_alpha()
    fond.fill(FOND)
    font = pygame.font.Font(None, int(45*WC))
    text = font.render(m, True, BLACK)
    fond.blit(text,(0,0))
    screen.blit(fond, (5*WC,5*WC))

#-----------------------------------------------------------------------------------------------------------------------

