------------------------------- À lire avant l'utilisation -------------------------------

Projet Sudoku - 2022

Ce projet a été mené dans le cadre de l'UEC informatique. Le but était de créer un progr-
amme qui permet à l'utilisateur de résoudre une grille de sudoku sur son ordinateur à pa-
rtir d'une grille présente sur une image et de lui apporter différentes aides.

Le projet s'est divisé en quatre grandes parties :
	- Interaction Homme - Machine
	- Traitement d'image
	- Reconnaissance de chiffres
	- Solveur

-------------------------------------- Instructions --------------------------------------

Pour lancer l'exécution du programme, il faut lancer le fichier "GUI.py"

Si toutes les librairies sont correctement installées, un explorateur de fichiers devrait
s'ouvrir. Vous pouvez alors sélectionner l'image d'une grille de sudoku. Les formats sup-
portés sont jpg et png.

L'interface peut mettre un peu de temps à apparaître. 

Vous pouvez ensuite jouer sur votre grille de sudoku numérisée et utiliser nos différentes
aides et modes de jeu.

Les modes de jeu :
	- Sans aides
	- Vérification des entrées : L'entrée correspondant à la bonne solution est indiquée
		en vert
	- Signalement des erreurs : L'entrée qui ne peut pas se trouver dans la case (car
		le chiffre est présent dans la ligne / colonne / bloc) est indiquée en rouge

Les aides :
	- Nombre de possibilités pour la case : Donne le nombre de chiffres possibles pour 
		la case sélectionnée (le joueur doit préalablement avoir cliqué sur une case)
	- Cases possibles pour le nombre entré : Donne tous les emplacements possibles pour 
		un nombre donné (le joueur doit préalablement avoir entré un nombre dans une
		case)
	- Afficher la solution : Affiche la solution (l'avancée du joueur est alors perdue)

------------------------------ Librairies python utilisées -------------------------------

À installer :
	-pygame	
	-opencv-python	
	-matplotlib	
	-numpy
	-torch (version 1.11.0)
	-torchvision (version 0.12.0)
	-scikit-image
	-scipy
	-PySat (python-sat)

Intégrées à python :
	-time
	-os
	-threading
	-tkinter
	

---------------------------------------- Auteurs -----------------------------------------

Groupe Gazon [Peip 2] :
	Eowyn Hallereau
	Emie Robin
	Alan Petit
	Chloé Mallet

Avec l'aide de :
	Nicolas Normand
	Jingwen Zhu
	Toinon Vigier
	Kevin Riou
	José Martinez

------------------------------------------------------------------------------------------


