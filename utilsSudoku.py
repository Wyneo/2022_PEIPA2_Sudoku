# fonction qui parcourt les cases sur la même ligne, la même colonne, et le même ensemble pour savoir si une valeur en (i,j) est autorisée
# Renvoie un booléen
def is_authorized(value, coordinates, G) :
    #parcours de la colonne
    for i in range(9) :
        if G[i][coordinates[1]] == value and i!=coordinates[0]:
            return False

    #parcours de la ligne
    for j in range(9) :
        if G[coordinates[0]][j] == value and j!=coordinates[1]:
            return False
    #parcours de la case
    for i in range(3) :
        for j in range(3) :
            if G[(coordinates[0]//3)*3+i][(coordinates[1]//3)*3+j] == value and (coordinates[0]//3)*3+i != i and (coordinates[1]//3)*3+j!=j:
                return False

    return True


