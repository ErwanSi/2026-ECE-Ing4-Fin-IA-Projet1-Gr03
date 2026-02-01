# Strategies

Ce projet implémente quatre stratégies d’exécution d’ordres : TWAP, VWAP, une optimisation sous contraintes (CP-SAT) et une approche par apprentissage par renforcement (RL).

---


## 1. TWAP (Time-Weighted Average Price)

**Principe**  
Le volume total à exécuter est réparti uniformément dans le temps. TWAP constitue une baseline simple et utile pour comparaison avec des stratégies plus sophistiquées.

**Formule**  
Pour un volume total (Q) et (N) tranches :

$$
x_t = Q / N
$$

(arrondi à l’entier et ajusté pour sommer exactement à (Q))

**Avantages**

* Très simple
* Prévisible

**Limites**

* Ignore totalement la liquidité du marché

---

## 2. VWAP (Volume-Weighted Average Price)

**Principe**  
Le volume est réparti proportionnellement au volume de marché observé ou historique, ce qui permet de tenir compte de la liquidité réelle du marché.

**Formule**  
Pour des volumes de marché (V_t) :

$$
x_t = Q \cdot \frac{V_t}{\sum_t V_t}
$$

Une contrainte de participation limite l’exécution :

$$
x_t \leq \alpha \cdot V_t
$$

avec $\alpha$ le taux de participation.

**Avantages**

* S’aligne sur la liquidité réelle

**Limites**

* Dépend des volumes observés

---

## 3. Optimisation sous contraintes (CP-SAT)

Cette approche rend l’exécution d’un ordre comme un **problème d’optimisation sous contraintes**.  
L’objectif principal est de répartir un **volume total à exécuter** sur plusieurs **tranches de temps**, tout en respectant les **limites de liquidité du marché** et en minimisant un **coût global** associé à l’exécution.

Le modèle est résolu à l’aide du **solveur CP-SAT de Google OR-Tools**, qui permet :
- de gérer des **variables entières**,
- d’imposer des **contraintes strictes**,
- et de définir une **fonction objectif quadratique**.

## Variables

$$
x_t \in \mathbb{N}^+ \quad \text{: volume exécuté à l’instant } t
$$

> Chaque tranche de temps \(t\) a un volume exécuté \(x_t\) qui doit être un entier positif.

## Données

- $Q$ : volume total à exécuter  
- $V_t$ : volume de marché observé à l’instant \(t\)  
- $\alpha$ : taux de participation maximal autorisé  

> Ces données définissent la quantité totale à exécuter, la liquidité disponible à chaque tranche, et la limite de participation par tranche.

## Contraintes

1. **Somme des volumes exécutés égale au volume total :**

$$
\sum_t x_t = Q
$$

> Cette contrainte assure que tout le volume prévu est exécuté.

2. **Borne de liquidité par tranche :**

$$
0 \le x_t \le \alpha \cdot V_t
$$

> Cela garantit que l’ordre ne dépasse pas une fraction raisonnable du volume du marché à chaque instant, limitant l’impact sur le marché.

## Cible VWAP

Le **volume théorique à exécuter pour suivre le VWAP** est donné par :

$$
x_t^{VWAP} = Q \cdot \frac{V_t}{\sum_k V_k}
$$

> Cette cible répartit le volume proportionnellement à la liquidité observée, permettant de suivre le **benchmark VWAP**.

## Fonction objectif

La fonction objectif cherche à **minimiser le coût global** en combinant deux composantes :

$$
\min \sum_t \left[ \lambda_{\text{impact}} \cdot x_t^2 + \lambda_{\text{track}} \cdot (x_t - x_t^{VWAP})^2 \right]
$$

- \(\lambda_{\text{impact}}\) : pondération de l’impact de marché (prévention des volumes trop concentrés)  
- \(\lambda_{\text{track}}\) : pondération du suivi du benchmark VWAP  

> La première partie pénalise les tranches trop importantes qui peuvent influencer le marché.  
> La seconde partie assure que l’exécution reste proche du VWAP.

## Sortie

Un **planning d’exécution optimal** :

$$
(x_1, x_2, \dots, x_N)
$$

> Chaque \(x_t\) indique le volume à exécuter à l’instant \(t\) pour atteindre un compromis optimal entre impact de marché et suivi du benchmark.

