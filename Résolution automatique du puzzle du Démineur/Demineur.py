import random
import numpy as np


MINE_VALUE = 9  # valeur utilisée pour représenter une mine


def creation_demineur(taille, bombes, x, y):
    """
    Crée une grille de démineur de taille (taille x taille) avec 'bombes' mines.
    Les coordonnées (x, y) correspondent à une case de départ qu'on garantit sans mine,
    ainsi que ses 8 voisines (zone 3x3).
    """
    demineur = np.zeros((taille, taille), dtype=int)

    i = 0
    while i < bombes:
        xr = random.randint(0, taille - 1)
        yr = random.randint(0, taille - 1)

        # Zone à protéger autour de (x, y) : la case et ses 8 voisines
        safe_zone = (
            (xr == x - 1 and yr == y - 1) or
            (xr == x - 1 and yr == y) or
            (xr == x - 1 and yr == y + 1) or
            (xr == x and yr == y - 1) or
            (xr == x and yr == y) or
            (xr == x and yr == y + 1) or
            (xr == x + 1 and yr == y - 1) or
            (xr == x + 1 and yr == y) or
            (xr == x + 1 and yr == y + 1)
        )

        if demineur[xr][yr] == 0 and not safe_zone:
            demineur[xr][yr] = MINE_VALUE
            i += 1

    return demineur


def ajout_nbr_casevide(demineur, i, j):
    """Incrémente la case (i, j) si ce n'est pas une mine."""
    if demineur[i][j] != MINE_VALUE:
        demineur[i][j] += 1


def mine_adjacent(demineur, taille):
    """
    Parcourt la grille et, pour chaque mine, incrémente les cases voisines.
    Au final, les cases non-mines contiennent le nombre de mines adjacentes.
    """
    for i in range(taille):
        for j in range(taille):
            if demineur[i][j] == MINE_VALUE:
                # Ligne du dessus
                if i != 0:
                    if j != 0:
                        ajout_nbr_casevide(demineur, i - 1, j - 1)
                    ajout_nbr_casevide(demineur, i - 1, j)
                    if j != taille - 1:
                        ajout_nbr_casevide(demineur, i - 1, j + 1)
                # Même ligne
                if j != 0:
                    ajout_nbr_casevide(demineur, i, j - 1)
                if j != taille - 1:
                    ajout_nbr_casevide(demineur, i, j + 1)
                # Ligne du dessous
                if i != taille - 1:
                    if j != 0:
                        ajout_nbr_casevide(demineur, i + 1, j - 1)
                    ajout_nbr_casevide(demineur, i + 1, j)
                    if j != taille - 1:
                        ajout_nbr_casevide(demineur, i + 1, j + 1)
    return demineur


def affichage(demineur, taille, mask):
    """
    Affichage texte (console) du plateau.
    Toujours disponible pour debug, même si tu utilises Pygame.
    mask[i][j] == 1 -> case cachée, 0 -> case révélée.
    """
    for i in range(taille):
        for k in range(taille * 4):
            print("_", end="")
            if k == taille * 4 - 1:
                print("_", end="\n")

        for j in range(taille):
            if mask[i][j]:
                print("#", end=" | ")
            else:
                print(demineur[i][j], end=" | ")
            if j == taille - 1:
                print("", end="\n")

    print()
    print()


def jeu(demineur, mask, taille, x, y):
    """
    Révèle la case (x, y). Si c'est une case vide (0), propage récursivement
    la révélation aux voisins (effet 'flood fill').
    Si c'est une mine, la case est simplement révélée.
    """
    # Si déjà révélée, ne fait rien
    if mask[x][y] == 0:
        return

    # Si la case est vide, on révèle et on propage
    if demineur[x][y] == 0:
        mask[x][y] = 0
        if x != 0:
            if y != 0:
                jeu(demineur, mask, taille, x - 1, y - 1)
            jeu(demineur, mask, taille, x - 1, y)
            if y != taille - 1:
                jeu(demineur, mask, taille, x - 1, y + 1)
        if y != 0:
            jeu(demineur, mask, taille, x, y - 1)
        if y != taille - 1:
            jeu(demineur, mask, taille, x, y + 1)
        if x != taille - 1:
            if y != 0:
                jeu(demineur, mask, taille, x + 1, y - 1)
            jeu(demineur, mask, taille, x + 1, y)
            if y != taille - 1:
                jeu(demineur, mask, taille, x + 1, y + 1)
    else:
        # Si ce n'est pas 0 (mine ou nombre), on révèle juste la case
        mask[x][y] = 0


def fin_jeu(demineur, mask, taille, bombe):
    """
    Vérifie l'état de la partie :
    - retourne 0 si la partie est terminée (perdu ou gagné),
    - retourne 1 si la partie continue.
    La détection du message 'gagné' / 'perdu' visuel est gérée par l'interface.
    """
    nb_mine = 0
    nb_demine = 0
    for i in range(taille):
        for j in range(taille):
            if demineur[i][j] == MINE_VALUE:
                if mask[i][j] == 0:
                    # Une mine a été révélée -> perdu
                    # (le message est géré côté interface)
                    return 0
                else:
                    nb_mine += 1

            if mask[i][j] == 1:
                nb_demine += 1

    # Si le nombre de cases encore cachées == nombre de mines -> gagné
    if nb_mine == nb_demine:
        return 0

    return 1
