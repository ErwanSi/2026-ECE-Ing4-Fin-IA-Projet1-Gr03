# Reinforcement Learning

L’apprentissage par renforcement (RL) permet de modéliser l’exécution d’un ordre comme un **processus décisionnel séquentiel**, où chaque décision influence l’évolution future du portefeuille et le coût d’exécution.  
Un agent interagit avec un **environnement de marché simulé**, observe l’état du marché et le volume restant, et apprend progressivement une **politique d’exécution optimale**.  
Cette approche est particulièrement utile lorsqu’on ne connaît pas à l’avance le profil de liquidité global, contrairement aux méthodes d’optimisation sous contraintes.

## Environnement

Le temps est discret et découpé en **N tranches temporelles**.  
À chaque instant $t$, l’agent choisit une quantité $a_t$ à exécuter, tout en respectant les contraintes de liquidité du marché et le volume restant à trader.

## État

L’état de l’agent à l’instant $t$ est défini par :

$$
s_t = (t, q_{remaining})
$$

- $t$ : indice de la tranche temporelle  
- $q_{remaining}$ : volume restant à exécuter

Cet état permet à l’agent de prendre des décisions en tenant compte à la fois du temps écoulé et du volume restant.

## Actions

L’action correspond à la fraction du **volume maximum autorisé** à l’instant $t$ :

$$
a_t \in \{0, 0.25, 0.5, 0.75, 1.0\} \cdot cap_t
$$

où $cap_t$ est la limite de participation maximale pour éviter un impact de marché trop important.

## Récompense

La récompense $r_t$ combine deux composantes : l’impact de marché et le tracking error par rapport au VWAP :

$$
r_t = - \left( \lambda_{impact} a_t^2 + \lambda_{track} (a_t - x_t^{VWAP})^2 \right)
$$

- $\lambda_{impact}$ : pondération de l’impact de marché (favorise l’exécution douce)  
- $\lambda_{track}$ : pondération du suivi du VWAP  

Une **pénalité terminale** est ajoutée si le volume total n’est pas exécuté en fin d’épisode, encourageant l’agent à terminer l’ordre.

## Algorithme

L’apprentissage utilise **Q-learning tabulaire** avec une politique $\epsilon$-greedy :

$$
Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') \right]
$$

- $\alpha$ : taux d’apprentissage  
- $\gamma$ : facteur de discount  
- $\epsilon$ : probabilité de choisir une action aléatoire pour explorer l’espace d’actions

## Sortie

Le résultat est une **politique d’exécution apprise**, testée via des rollouts en mode greedy, qui indique la fraction optimale du volume à exécuter à chaque tranche temporelle, en minimisant à la fois l’impact de marché et le tracking error.

**Avantages**

* Adaptatif
* Approche en ligne

**Limites**

* Dépend de l’entraînement et des hyperparamètres