# 11. Résolution automatique du puzzle du Démineur


## Description du problème et contexte
Le jeu du Démineur se résout automatiquement en modélisant le problème sous forme de CSP. Chaque case inconnue de la grille est représentée par une variable booléenne indiquant la présence ou non d'une mine. Pour chaque case ouverte, le chiffre affiché impose que le nombre de mines dans son voisinage corresponde exactement à cette valeur. La propagation de contraintes permet de déduire systématiquement quelles cases sont sûres et lesquelles contiennent une mine, bien que le problème soit NP-complet dans sa version générale.