import pygame
import os
os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.init()
info = pygame.display.Info()
window_W = info.current_w - 10                                                                                                  #Taille fenêtre de jeu
WC = window_W / 1500

#------------------------------------------Boutons----------------------------------------------------------------------
B0 = pygame.image.load("TemplateB/Boutons/B0.png")
B1 = pygame.image.load("TemplateB/Boutons/B1.png")
B2 = pygame.image.load("TemplateB/Boutons/B2.png")
B3 = pygame.image.load("TemplateB/Boutons/B3.png")
B4 = pygame.image.load("TemplateB/Boutons/B4.png")
B5 = pygame.image.load("TemplateB/Boutons/B5.png")

B0 = pygame.transform.scale(B0, (117*WC, 114*WC))
B1 = pygame.transform.scale(B1, (117*WC, 114*WC))
B2 = pygame.transform.scale(B2, (117*WC, 114*WC))
B3 = pygame.transform.scale(B3, (117*WC, 114*WC))
B4 = pygame.transform.scale(B4, (117*WC, 114*WC))
B5 = pygame.transform.scale(B5, (117*WC, 114*WC))

#------------------------------------Template + grille------------------------------------------------------------------
grid_image=pygame.image.load("TemplateB/grilleVide2.png")
grid_image = pygame.transform.scale(grid_image, (509*WC, 509*WC))

template = pygame.image.load("TemplateB/TemplateTotale.png")
template = pygame.transform.scale(template, (1500 * WC, 700 * WC))