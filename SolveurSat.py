import pysat

G_init = [[5,0,0, 0,2,0, 8,0,9],
          [0,4,1, 8,0,0, 0,6,0],
          [0,0,2, 6,0,9, 3,0,0],
          [0,0,7, 5,0,8, 0,1,0],
          [0,9,0, 0,4,0, 5,0,7],
          [4,5,0, 0,0,1, 0,2,0],
          [6,0,4, 0,1,0, 0,0,2],
          [0,1,0, 7,0,0, 0,5,4],
          [8,0,0, 0,6,2, 1,0,0]]

CNF=[]
for i in range (9):
    for j in range (9):
        if G_init[i][j]!=0:
            CNF.append(((1,(i,j,G_init[i][j]))))
            for k in range(9):
                if k!=i:
                    CNF.append(((0,(i,j,G_init[i][j])),(0,(k,j,G_init[i][j]))))
            for l in range(9):
                if l!=j:
                    CNF.append(((0,(i,j,G_init[i][j])),(0,(i,l,G_init[i][j]))))
            CaseI=i//3*3
            CaseJ=j//3*3
            for a in range(0,3):
                for b in range(0,3):
                    if a!=i or b!=j:
                        CNF.append(((0,(i,j,G_init[i][j])),(0,(CaseI+a,j, G_init[i][j]))))
                        CNF.append(((0,(i,j,G_init[i][j])),(0,(i,CaseJ+b,G_init[i][j]))))

clé = 0
PvS = {}
SvP = {}
for i in range(9):
    for j in range (9):
        for k in range(9):
            caseSvP = {"T":(i,j,k+1)}
            SvP[clé]=caseSvP
            casePvS = {"C":clé}
            PvS[(i,j,k+1)]=casePvS
            clé=clé+1


CNFSolveur = []
for i in range(len(CNF)):
    p=[]
    for j in range(len(CNF[i])):
        if CNF[i][j]==0: t = -1
        else : t= 1
        p.append(PvS[CNF[i][j]]["C"]*t)
    CNFSolveur.append(p)
print(CNFSolveur)