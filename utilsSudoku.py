# fonction qui parcourt les cases sur la même ligne, la même colonne, et le même ensemble pour savoir si une valeur en (i,j) est autorisée
# Renvoie un booléen
def is_authorized(value, coordinates, G) :
    #parcours de la colonne
    for i in range(9) :
        if G[i][coordinates[1]] == value :
            return False

    #parcours de la ligne
    for j in range(9) :
        if G[coordinates[0]][j] == value:
            return False
    #parcours de la case
    for i in range(3) :
        for j in range(3) :
            if G[(coordinates[0]//3)*3+i][(coordinates[1]//3)*3+j] == value:
                return False

    return True


