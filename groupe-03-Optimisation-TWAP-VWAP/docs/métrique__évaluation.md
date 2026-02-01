# Métriques d’évaluation

Cette section décrit les expériences menées pour comparer les différentes stratégies d’exécution et analyser leurs performances dans des conditions variées.

## Stratégies comparées

Les stratégies évaluées comprennent :

- **TWAP** (répartit uniformément le volume sur le temps)
- **VWAP** (répartit proportionnellement aux volumes de marché)
- **Optimisation sous contraintes (CP-SAT / CSP)** (compromis entre impact de marché et tracking error)
- **Agent de reinforcement learning (RL)** (apprend et décide des volumes à exécuter tranche par tranche)

## Scénarios

Les tests sont réalisés sur différents types de profils de volume :

- **Profils synthétiques** : générés pour tester des comportements extrêmes et des situations contrôlées.
- **Profils intraday réels** : volumes de marché réels afin d’évaluer les performances dans des conditions réalistes.

## Impact de marché

L’impact de marché est approximé par un **proxy quadratique normalisé par la liquidité du marché** :

$$
Impact(x) = \sum_{t=1}^{N} \frac{x_t^2}{V_t + \varepsilon}
$$

où :
- $x_t$ est le volume exécuté à la tranche $t$,
- $V_t$ est le volume de marché observé sur la tranche $t$,
- $\varepsilon$ est une constante très petite évitant la division par zéro.

Cette métrique pénalise davantage les exécutions agressives dans les tranches **peu liquides**, et pénalise moins lorsque le marché est **profond et liquide**.

## Erreur de tracking VWAP

L’erreur de tracking mesure la capacité de la stratégie à suivre une cible VWAP en volume (profil “idéal”) :

$$
Tracking_{L2}(x) = \sum_{t=1}^{N} \left(x_t - x_t^{VWAP}\right)^2
$$

où $x_t^{VWAP}$ est le volume cible pour suivre le VWAP, calculé à partir des volumes de marché par une répartition proportionnelle, puis **discrétisé** afin de garantir que :

$$
\sum_{t=1}^{N} x_t^{VWAP} = Q
$$

## Contraintes d’exécution

La stratégie optimisée est soumise aux contraintes opérationnelles suivantes :

### 1. Contrainte de complétion (ordre total)

Le volume total exécuté doit être égal à la quantité cible :

$$
\sum_{t=1}^{N} x_t = Q
$$

### 2. Non-négativité

Le volume exécuté à chaque tranche doit être positif ou nul :

$$
x_t \ge 0 \quad \forall t
$$

### 3. Contrainte de participation maximale (participation rate)

Chaque tranche est limitée à une fraction maximale du volume de marché :

$$
x_t \le \alpha \, V_t \quad \forall t
$$

où $\alpha \in (0,1]$ est le **taux de participation maximal autorisé**.

### 4. Capacité maximale par tranche (override / max par slice)

En plus du taux de participation global, il est possible d’imposer un plafond strict sur certaines tranches :

$$
x_t \le \overline{cap}_t \quad \forall t
$$

La contrainte effectivement appliquée dans le solveur est donc :

$$
x_t \le \min\left(\alpha V_t,\; \overline{cap}_t\right)
$$

## Analyse

Les résultats expérimentaux mettent en évidence les caractéristiques et limites de chaque approche :

- **TWAP** : simple à mettre en œuvre, mais rigide, ne tient pas compte de la liquidité.
- **VWAP** : plus aligné sur le marché, mais suppose une connaissance préalable ou une estimation fiable des volumes futurs. 
- **CP-SAT / CSP** : atteint un compromis contrôlé entre impact et tracking, tout en respectant les contraintes de participation et de capacité par tranche.
- **RL** : offre une flexibilité adaptative et peut s’ajuster en ligne aux variations du marché.

Ces expériences permettent de comparer quantitativement les stratégies en fonction de leurs objectifs d’exécution et de leurs contraintes opérationnelles.
