import pysat
from pysat.formula import CNF
from pysat.solvers import Solver


'''Fonction SolveurSAT : prend en compte une grille de Sudoku non résolue et la résoud à l'aide de SAT
    Entrée : tableau de tableaux représentant une grille de Sudoku non résolue
    Sortie : tableau de la même forme représentant la grille résolue'''
def SolveurSat(G_init):

    #-------------------------------------------------Remplissage du CNF------------------------------------------------
    CNFA = []
    for i in range(9):
        for j in range(9):
            CNFA.append(((1, (i, j, 1)), (1, (i, j, 2)), (1, (i, j, 3)), (1, (i, j, 4)), (1, (i, j, 5)), (1, (i, j, 6)), (1, (i, j, 7)), (1, (i, j, 8)), (1, (i, j, 9))))
            for v in range(1, 10):
                for n in range(1, 10):
                    if n != v:
                        CNFA.append(((0, (i, j, v)), (0, (i, j, n))))
            k = G_init[i][j]
            CaseI = i // 3 * 3
            CaseJ = j // 3 * 3
            if k != 0:
                CNFA.append(((1, (i, j, k)), (1, (i, j, k))))
                for m in range(9):
                    if m != i:
                        CNFA.append(((0, (i, j, k)), (0, (m, j, k))))
                for l in range(9):
                    if l != j:
                        CNFA.append(((0, (i, j, k)), (0, (i, l, k))))
                for a in range(0, 3):
                    for b in range(0, 3):
                        if (CaseI+a != i) and (CaseJ+b != j):
                            CNFA.append(((0, (i, j, k)), (0, (CaseI+a, CaseJ+b, k))))

            for v in range(1, 10):
                for b in range(9):
                    if b != j:
                        CNFA.append(((0, (i, j, v)), (0, (i, b, v))))
                for d in range(9):
                    if d != i:
                        CNFA.append(((0, (i, j, v)), (0, (d, j, v))))
                for a in range(0, 3):
                    for b in range(0, 3):
                        if (CaseI+a != i) and (CaseJ+b != j):
                            CNFA.append(((0, (i, j, v)), (0, (CaseI+a, CaseJ+b, v))))

    #---------------------------------------------Création des dictionnaires--------------------------------------------
    clé = 1
    PvS = {}
    SvP = {}
    for i in range(9):
        for j in range(9):
            for k in range(9):
                caseSvP = {"Tuple": (i, j, k+1)}
                SvP[clé] = caseSvP
                casePvS = {"Clé": clé}
                PvS[(i, j, k+1)] = casePvS
                clé = clé+1

    CNFASolveur = []

    #---------------------------------------Conversion des tuples du CNF en entiers-------------------------------------

    for i in range(len(CNFA)):
        p = []
        if (CNFA[i][0][0]) == 0:
            t = -1
            p.append(((PvS[CNFA[i][0][1]]["Clé"] * t), (PvS[CNFA[i][1][1]]["Clé"] * t)))
        else:
            t = 1
            if len(CNFA[i]) == 2:
                p.append(((PvS[CNFA[i][0][1]]["Clé"]*t), (PvS[CNFA[i][1][1]]["Clé"]*t)))
            else :
                p.append(((PvS[CNFA[i][0][1]]["Clé"] * t), (PvS[CNFA[i][1][1]]["Clé"] * t), (PvS[CNFA[i][2][1]]["Clé"] * t), (PvS[CNFA[i][3][1]]["Clé"] * t), (PvS[CNFA[i][4][1]]["Clé"] * t), (PvS[CNFA[i][5][1]]["Clé"] * t), (PvS[CNFA[i][6][1]]["Clé"] * t), (PvS[CNFA[i][7][1]]["Clé"] * t), (PvS[CNFA[i][8][1]]["Clé"] * t)))

        CNFASolveur.append(p[0])

    #-------------------------------------------------Résolution par SAT------------------------------------------------
    #print('nombre de clauses : ', len(CNFASolveur))
    cnf = CNF(from_clauses=CNFASolveur)

    with Solver(bootstrap_with=cnf) as solver:
        #print('formula is', f'{"s" if solver.solve() else "uns"}atisfiable')
        if solver.solve():
            sol = solver.get_model()
        else :
            print("Aucune solution n'a été trouvée")
            return None
    soluce = []
    for i in sol:
        if i >= 0:
            soluce.append(SvP[i]['Tuple'])

    G_final = [[0 for m in range(9)] for n in range(9)]
    for i in range(len(soluce)):
        G_final[soluce[i][0]][soluce[i][1]] = soluce[i][2]

    return (G_final)

