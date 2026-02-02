"""
Module d'assistance LLM (Large Language Model) - STUB PÉDAGOGIQUE

Ce module est un STUB (désactivé par défaut) qui démontre COMMENT
un LLM pourrait être intégré au solveur Wordle CSP.

IMPORTANT:
- Ce module ne nécessite AUCUNE clé API
- Il s'agit d'une démonstration pédagogique
- Les fonctions retournent des réponses simulées

PERSPECTIVES D'INTÉGRATION LLM:

1. Explication du raisonnement:
   - Le LLM pourrait expliquer pourquoi un mot est suggéré
   - Vulgariser les concepts CSP pour l'utilisateur
   - Fournir du contexte linguistique

2. Proposition de mots:
   - Le LLM pourrait proposer des mots basés sur:
     * Sémantique (mots liés au contexte)
     * Phonétique (mots qui sonnent similaires)
     * Fréquence dans la langue

3. Hybridation neuro-symbolique:
   - IA symbolique (CSP): garantit la cohérence logique
   - IA neuronale (LLM): apporte l'intuition et le contexte
   - Combinaison: meilleure performance + explicabilité

LIMITATIONS:
- Les LLM ne garantissent pas la cohérence logique
- Le CSP reste nécessaire pour le filtrage rigoureux
- Le LLM est un complément, pas un remplacement
"""

from typing import List, Dict
import random


# Configuration (désactivé par défaut)
LLM_ENABLED = False


def is_llm_available() -> bool:
    """
    Vérifie si un LLM est disponible.

    Returns:
        False (stub pédagogique)
    """
    return LLM_ENABLED


def explain_suggestion(word: str, candidates: List[str], constraints: List) -> str:
    """
    Génère une explication humaine pour une suggestion de mot.

    Dans une vraie implémentation, le LLM recevrait:
    - Le mot suggéré
    - Les contraintes actuelles
    - L'historique des tentatives

    Et retournerait une explication en langage naturel.

    Args:
        word: mot suggéré
        candidates: liste des candidats restants
        constraints: contraintes CSP appliquées

    Returns:
        Explication textuelle (simulée)

    Exemple de sortie:
        "Je suggère GERER car:
         - Il contient 'E' qui apparaît dans 80% des candidats
         - Il utilise la lettre 'R' en position 2, ce qui élimine 40% des mots restants
         - C'est un mot fréquent en français, augmentant les chances"
    """
    if not LLM_ENABLED:
        return _simulate_explanation(word, candidates)

    # Dans une vraie implémentation:
    # prompt = f"Explique pourquoi suggérer '{word}' sachant que..."
    # response = llm_api.complete(prompt)
    # return response

    return _simulate_explanation(word, candidates)


def _simulate_explanation(word: str, candidates: List[str]) -> str:
    """Simule une explication LLM."""
    reasons = [
        f"Le mot '{word}' contient des lettres fréquentes dans les {len(candidates)} candidats restants",
        f"Ce choix maximise l'entropie, permettant d'éliminer efficacement les possibilités",
        f"'{word}' est un mot courant en français, augmentant la probabilité de succès"
    ]

    explanation = f"💡 Je suggère {word} car:\n"
    for reason in random.sample(reasons, 2):
        explanation += f"  - {reason}\n"

    return explanation


def suggest_word_with_context(candidates: List[str], context: str = None) -> Dict:
    """
    Suggère un mot en tenant compte du contexte linguistique.

    Un LLM pourrait proposer des mots basés sur:
    - Le contexte thématique ("mots liés à la nature")
    - La structure phonétique
    - Les associations sémantiques

    Args:
        candidates: liste des candidats CSP
        context: contexte optionnel (ex: "animaux", "nature")

    Returns:
        Dictionnaire avec suggestion et explication
    """
    if not candidates:
        return {
            'word': None,
            'explanation': "Aucun candidat disponible"
        }

    # Dans une vraie implémentation:
    # Le LLM recevrait le contexte et proposerait un mot
    # Puis le CSP validerait que ce mot est dans les candidats

    word = random.choice(candidates)
    explanation = _simulate_explanation(word, candidates)

    return {
        'word': word,
        'explanation': explanation,
        'context_used': context is not None
    }


def explain_csp_concept(concept: str) -> str:
    """
    Explique un concept CSP en langage naturel.

    Utile pour la vulgarisation pédagogique.

    Args:
        concept: nom du concept ('variable', 'domaine', 'contrainte', 'arc-consistency')

    Returns:
        Explication en français

    Exemple:
        >>> explain_csp_concept('arc-consistency')
        "L'arc-consistency (cohérence d'arc) est une technique de propagation
        de contraintes qui élimine les valeurs incompatibles dans les domaines..."
    """
    explanations = {
        'variable': """
        Une VARIABLE dans un CSP représente un élément à déterminer.
        Dans Wordle, on peut modéliser chaque position (1-5) comme une variable,
        ou considérer une seule variable 'mot' dont le domaine est le dictionnaire.
        """,

        'domaine': """
        Le DOMAINE d'une variable est l'ensemble des valeurs possibles.
        Dans Wordle, le domaine initial est l'ensemble de tous les mots de 5 lettres.
        Après chaque feedback, le domaine se réduit.
        """,

        'contrainte': """
        Une CONTRAINTE limite les valeurs possibles des variables.
        Dans Wordle, les contraintes sont:
        - Lettres vertes: position exacte
        - Lettres jaunes: présence mais mauvaise position
        - Lettres grises: absence
        """,

        'arc-consistency': """
        L'ARC-CONSISTENCY (cohérence d'arc) est une technique de propagation.
        Elle élimine les valeurs du domaine qui ne peuvent satisfaire aucune solution.
        Dans Wordle, on filtre tous les mots incompatibles avec les feedbacks reçus.
        """,

        'heuristique': """
        Une HEURISTIQUE est une règle intuitive pour guider la recherche.
        Dans Wordle, nos heuristiques choisissent le mot qui:
        - Maximise les lettres fréquentes (heuristique fréquence)
        - Maximise l'information gagnée (heuristique entropie)
        """
    }

    return explanations.get(
        concept,
        f"Concept '{concept}' non documenté dans ce stub."
    )


def hybrid_neuro_symbolic_suggestion(
    symbolic_suggestion: str,
    candidates: List[str],
    use_llm_boost: bool = False
) -> Dict:
    """
    Démontre une approche hybride neuro-symbolique.

    Principe:
    1. Le système symbolique (CSP) filtre les candidats valides
    2. Le système neuronal (LLM) ordonne les candidats par pertinence
    3. La combinaison donne le meilleur des deux mondes

    Args:
        symbolic_suggestion: suggestion du CSP
        candidates: candidats valides selon le CSP
        use_llm_boost: utiliser le LLM pour ré-ordonner

    Returns:
        Suggestion finale avec explication
    """
    result = {
        'symbolic_suggestion': symbolic_suggestion,
        'final_suggestion': symbolic_suggestion,
        'llm_used': False,
        'explanation': "Suggestion purement symbolique (CSP)"
    }

    if use_llm_boost and LLM_ENABLED:
        # Dans une vraie implémentation:
        # 1. Demander au LLM de scorer chaque candidat
        # 2. Combiner avec le score CSP
        # 3. Retourner le meilleur

        result['llm_used'] = True
        result['explanation'] = "Suggestion hybride: CSP + LLM"

    return result


def generate_educational_summary(solver_stats: Dict) -> str:
    """
    Génère un résumé pédagogique de la résolution.

    Utile pour expliquer le processus à un étudiant.

    Args:
        solver_stats: statistiques du solveur CSP

    Returns:
        Résumé en français
    """
    summary = f"""
    📚 RÉSUMÉ PÉDAGOGIQUE

    État initial:
    - {solver_stats.get('initial_candidates', 0)} mots dans le dictionnaire

    Après résolution:
    - {solver_stats.get('constraints_applied', 0)} contraintes appliquées
    - {solver_stats.get('current_candidates', 0)} candidats restants
    - Taux de réduction: {solver_stats.get('reduction_rate', 0):.1%}

    Principe CSP:
    Chaque feedback ajoute une contrainte qui élimine les mots incompatibles.
    C'est une approche symbolique: logique pure, pas de "devinette".

    L'IA exploratoire optimise le choix des mots pour minimiser le nombre
    de coups nécessaires.
    """

    return summary.strip()


# Exemple d'utilisation pédagogique
if __name__ == '__main__':
    print("Module LLM Assist - Stub pédagogique")
    print("=" * 60)
    print(f"LLM activé: {is_llm_available()}")
    print()

    # Démonstration des explications
    print("Explication du concept 'contrainte':")
    print(explain_csp_concept('contrainte'))
    print()

    # Simulation de suggestion
    candidates = ['ARBRE', 'AUTRE', 'AITRE']
    result = suggest_word_with_context(candidates)
    print(f"Suggestion simulée: {result['word']}")
    print(result['explanation'])
