import numpy as np


# Cette fonction permet d'avoir un plateau de jeu comme le joueur voit
def plateau(demineur, mask, taille):
    plat = np.zeros(demineur.shape, dtype=int)
    for x in range(taille):
        for y in range(taille):
            if mask[x][y] == 1:
                plat[x][y] = -1
            else:
                plat[x][y] = demineur[x][y]
    return plat


def case_decouverte(plat, x, y):
    if plat[x][y] == -1:
        return 0
    return 1


def bordure(plat, taille):
    for i in range(taille):
        for j in range(taille):
            if plat[i][j] >= 0:
                # Ligne du dessus
                if i != 0:
                    if j != 0:
                        if not case_decouverte(plat, i - 1, j - 1):
                            plat[i - 1][j - 1] = -2
                    if not case_decouverte(plat, i - 1, j):
                        plat[i - 1][j] = -2
                    if j != taille - 1:
                        if not case_decouverte(plat, i - 1, j + 1):
                            plat[i - 1][j + 1] = -2
                # Même ligne
                if j != 0:
                    if not case_decouverte(plat, i, j - 1):
                        plat[i][j - 1] = -2
                if j != taille - 1:
                    if not case_decouverte(plat, i, j + 1):
                        plat[i][j + 1] = -2
                # Ligne du dessous
                if i != taille - 1:
                    if j != 0:
                        if not case_decouverte(plat, i + 1, j - 1):
                            plat[i + 1][j - 1] = -2
                    if not case_decouverte(plat, i + 1, j):
                        plat[i + 1][j] = -2
                    if j != taille - 1:
                        if not case_decouverte(plat, i + 1, j + 1):
                            plat[i + 1][j + 1] = -2

    return plat


def liste_bordure(plat):
    variables = []
    for i in range(len(plat)):
        for j in range(len(plat)):
            if plat[i][j] == -2:
                variables.append((i, j))
    return variables


def voisins(x, y, taille):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < taille and 0 <= ny < taille:
                yield nx, ny


def contraintes(plat):
    taille = len(plat)
    contraintes = []

    for i in range(taille):
        for j in range(taille):
            if plat[i][j] >= 0:  # case découverte
                inconnues = []
                for nx, ny in voisins(i, j, taille):
                    if plat[nx][ny] == -2:
                        inconnues.append((nx, ny))

                if inconnues:
                    contraintes.append((inconnues, plat[i][j]))

    return contraintes


def contraintes_valides(contraintes, assignation):
    for cases, total in contraintes:
        somme = 0
        libres = 0

        for c in cases:
            if c in assignation:
                somme += assignation[c]
            else:
                libres += 1

        if somme > total:
            return False
        if somme + libres < total:
            return False

    return True


def backtracking(variables, contraintes, assignation, stats):
    if len(assignation) == len(variables):
        stats["total"] += 1
        for v in assignation:
            if assignation[v] == 1:
                stats["mines"][v] += 1
        return

    v = variables[len(assignation)]

    for valeur in [0, 1]:
        assignation[v] = valeur
        if contraintes_valides(contraintes, assignation):
            backtracking(variables, contraintes, assignation, stats)
        del assignation[v]


def probabilites(plat):
    variables = liste_bordure(plat)
    cons = contraintes(plat)

    stats = {
        "total": 0,
        "mines": {v: 0 for v in variables}
    }

    backtracking(variables, cons, {}, stats)

    probs = {}
    for v in variables:
        probs[v] = stats["mines"][v] / stats["total"]

    return probs


def meilleure_case(probs):
    for c, p in probs.items():
        if p == 0:
            return c
    return min(probs, key=probs.get)


def Resolution(demineur, mask, taille, bombe):
    plat = plateau(demineur, mask, taille)
    bordure(plat, taille)
    probs = probabilites(plat)
    return meilleure_case(probs)
