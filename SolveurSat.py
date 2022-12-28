import pysat
from pysat.formula import CNF
from pysat.solvers import Solver

'''Fonction SolveurSAT : prend en compte une grille de Sudoku non résolue et la résoud à l'aide de SAT
    Entrée : tableau de tableaux représentant une grille de Sudoku non résolue
    Sortie : tableau de la même forme représentant la grille résolue'''

def SolveurSat(G_init):

    # ---------------------------------------------Création des dictionnaires--------------------------------------------
    clé = 1
    PvS = {}
    SvP = {}
    for i in range(9):
        for j in range(9):
            for k in range(9):
                caseSvP = {"Tuple": (i, j, k + 1)}
                SvP[clé] = caseSvP
                casePvS = {"Clé": clé}
                PvS[(i, j, k + 1)] = casePvS
                clé = clé + 1

    #-------------------------------------------------Remplissage du CNF------------------------------------------------
    CNFA = []
    for i in range(9):
        for j in range(9):
            CNFA.append((PvS[(i, j, 1)]["Clé"],PvS[(i, j, 2)]["Clé"],PvS[(i, j, 3)]["Clé"],PvS[(i, j, 4)]["Clé"],PvS[(i, j, 5)]["Clé"],PvS[(i, j, 6)]["Clé"],PvS[(i, j, 7)]["Clé"],PvS[(i, j, 8)]["Clé"],PvS[(i, j, 9)]["Clé"],))
            for v in range(1, 10):
                for n in range(1, 10):
                    if n != v:
                        CNFA.append((PvS[(i, j, v)]["Clé"]*(-1), PvS[(i, j, n)]["Clé"]*(-1)))
            k = G_init[i][j]
            CaseI = i // 3 * 3
            CaseJ = j // 3 * 3
            if k != 0:
                CNFA.append((PvS[(i, j, k)]["Clé"], PvS[(i, j, k)]["Clé"]))
                for m in range(9):
                    if m != i:
                        CNFA.append((PvS[(i, j, k)]["Clé"]*(-1), PvS[(m, j, k)]["Clé"]*(-1)))
                for l in range(9):
                    if l != j:
                        CNFA.append((PvS[(i, j, k)]["Clé"]*(-1), PvS[(i, l, k)]["Clé"]*(-1)))
                for a in range(0, 3):
                    for b in range(0, 3):
                        if (CaseI+a != i) and (CaseJ+b != j):
                            CNFA.append((PvS[(i, j, k)]["Clé"]*(-1), PvS[(CaseI+a, CaseJ+b, k)]["Clé"]*(-1)))

            for v in range(1, 10):
                for b in range(9):
                    if b != j:
                        CNFA.append((PvS[(i, j, v)]["Clé"]*(-1), PvS[(i, b, v)]["Clé"]*(-1)))
                for d in range(9):
                    if d != i:
                        CNFA.append((PvS[(i, j, v)]["Clé"]*(-1), PvS[(d, j, v)]["Clé"]*(-1)))
                for a in range(0, 3):
                    for b in range(0, 3):
                        if (CaseI+a != i) and (CaseJ+b != j):
                            CNFA.append((PvS[(i, j, v)]["Clé"]*(-1), PvS[(CaseI+a, CaseJ+b, v)]["Clé"]*(-1)))

    #-------------------------------------------------Résolution par SAT------------------------------------------------
    cnf = CNF(from_clauses=CNFA)
    with Solver(bootstrap_with=cnf) as solver:
        if solver.solve():
            sol = solver.get_model()
        else :
            return None
    soluce = []
    for i in sol:
        if i >= 0:
            soluce.append(SvP[i]['Tuple'])

    G_final = [[0 for m in range(9)] for n in range(9)]
    for i in range(len(soluce)):
        G_final[soluce[i][0]][soluce[i][1]] = soluce[i][2]

    return (G_final)