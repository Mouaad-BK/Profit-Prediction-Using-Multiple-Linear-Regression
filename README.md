# Prédiction des Profits - Régression Linéaire Multiple

## Description du projet

Ce projet utilise des techniques de **Machine Learning** pour prédire les profits d'une entreprise à partir de plusieurs caractéristiques financières. L'objectif est d'analyser les données historiques (par exemple, les dépenses en R&D, en marketing, etc.) et d'utiliser une **régression linéaire multiple** pour estimer les profits futurs. Ce projet inclut un prétraitement des données, de l'encodage des variables, de la réduction de dimensionnalité et du calcul de la performance du modèle.

L'application web développée avec **Flask** permet à l'utilisateur de télécharger un fichier CSV avec des données financières, effectuer les transformations nécessaires et obtenir une prédiction des profits.

---

## Fonctionnalités

- **Chargement des données :** L'utilisateur peut télécharger un fichier CSV contenant des données d'entrée.
- **Nettoyage des données :** Gestion des valeurs manquantes (remplissage par la moyenne, médiane, ou suppression des lignes).
- **Encodage des variables :** Transformation des variables catégorielles en valeurs numériques à l'aide de méthodes comme One-Hot Encoding, Label Encoding, ou Binary Encoding.
- **Normalisation des données :** Application de techniques de normalisation (Min-Max Scaling, Standardization, etc.).
- **Réduction de la dimensionnalité (ACP) :** Utilisation de l'Analyse en Composantes Principales (PCA) pour réduire la complexité des données.
- **Modèle de régression linéaire multiple :** Entraînement d'un modèle de régression linéaire pour prédire les profits.
- **Visualisation des données :** Affichage de graphiques (matrice de corrélation, composantes principales, etc.).
- **Prédiction des profits :** Interface pour saisir de nouvelles données et obtenir une prédiction en temps réel.

---

## Technologies Utilisées

- **Python** - Langage de programmation principal.
- **Flask** - Framework web pour construire l'application.
- **Pandas** - Pour la manipulation et l'analyse des données.
- **NumPy** - Pour les calculs numériques.
- **Matplotlib** et **Seaborn** - Pour la visualisation des données.
- **Scikit-learn** - Pour l'entraînement du modèle de régression linéaire multiple et le prétraitement des données.
- **HTML/CSS/JavaScript** - Pour l'interface utilisateur et l'interactivité.

---