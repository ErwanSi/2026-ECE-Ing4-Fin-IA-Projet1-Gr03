# Data Pipeline

Les données de marché sont utilisées pour construire des **profils de liquidité réalistes**, servant de base aux stratégies VWAP, CP-SAT et Reinforcement Learning (RL). Ces profils permettent de simuler des conditions de marché proches du réel et d’évaluer la performance des stratégies.

Dans ce projet, les données sont récupérées sous forme de **snapshots intraday**, fournissant les prix et volumes échangés à intervalles réguliers.

## Source

Yahoo Finance (via l’API `yfinance`) est utilisée pour récupérer automatiquement les données historiques intraday.

## Données utilisées

- **Prix de clôture intraday** : $P_t$ à chaque tranche $t$  
- **Volumes échangés par intervalle** : $V_t$ à chaque tranche $t$  

Ces données sont essentielles pour construire les profils VWAP et déterminer les limites de participation au marché.

## Prétraitement

- **Sélection d’une fenêtre intraday** pertinente pour l’exécution.  
- **Agrégation des volumes par tranche de temps** :

$$
V_t^{agg} = \sum_{i \in tranche~t} V_i
$$

- **Normalisation des volumes** si nécessaire :

$$
V_t^{norm} = \frac{V_t^{agg}}{\sum_t V_t^{agg}}
$$

Cette normalisation permet de comparer différents actifs et d’ajuster les contraintes de participation.

## Rôle dans le projet

Les volumes et prix intraday servent à :

- **VWAP** : calculer les volumes cibles proportionnels à la liquidité observée.

$$
x_t^{VWAP} = Q_{total} \cdot V_t^{norm}
$$

- **CP-SAT** : définir les bornes de participation par tranche et le volume total à exécuter.

$$
a_t^{min} = 0, \quad a_t^{max} = \alpha \cdot V_t^{agg}
$$

- **RL** : fournir la dynamique de l’environnement simulé pour l’apprentissage de l’agent.

$$
state_t = (t, q_{remaining}), \quad reward_t = - \left( \lambda_{impact} \cdot a_t^2 + \lambda_{track} \cdot (a_t - x_t^{VWAP})^2 \right)
$$

> Aucune information future n’est utilisée dans les stratégies en ligne, garantissant une exécution réaliste et non anticipative.
