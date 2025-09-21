# Sudoku

### Mots-clés : IA, Traitement d'images, Solveur SAT, Interaction homme-machine, Python

### Projet : Créer un programme qui permet à l'utilisateur de résoudre une grille de sudoku sur son ordinateur à partir d'une grille présente sur une image et de lui apporter différentes aides.

Le projet s'est divisé en quatre grandes parties :
	- Interaction Homme - Machine
	- Traitement d'image
	- Reconnaissance de chiffres
	- Solveur

Pour lancer l'exécution du programme, il faut lancer le fichier "GUI.py". Si toutes les librairies sont correctement installées, un explorateur de fichiers devrait s'ouvrir. Vous pouvez alors sélectionner l'image d'une grille de sudoku. Les formats supportés sont jpg et png. L'interface peut mettre un peu de temps à apparaître. 

Vous pouvez ensuite jouer sur votre grille de sudoku numérisée et utiliser nos différentes aides et modes de jeu.  
Les modes de jeu :  
- Sans aides
- Vérification des entrées : L'entrée correspondant à la bonne solution est indiquée en vert.
- Signalement des erreurs : L'entrée qui ne peut pas se trouver dans la case (car le chiffre est présent dans la ligne / colonne / bloc) est indiquée en rouge.

Les aides :
- Nombre de possibilités pour la case : Donne le nombre de chiffres possibles pour la case sélectionnée (le joueur doit préalablement avoir cliqué sur une case).
- Cases possibles pour le nombre entré : Donne tous les emplacements possibles pour un nombre donné (le joueur doit préalablement avoir entré un nombre dans une case).
- Afficher la solution : Affiche la solution (l'avancée du joueur est alors perdue).  

Librairies python utilisées : pygame, opencv-python, matplotlib, numpy, torch (version 1.11.0), torchvision (version 0.12.0), scikit-image, scipy, PySat (python-sat), time, os, threading, tkinter

Projet réalisé avec Mm. Robin, M. Petit, Mm. Mallet.  
Remerciements à M. Normand, Mm. Zhu, Mm. Vigier, M. Riou et M. Martinez pour leur aide.



