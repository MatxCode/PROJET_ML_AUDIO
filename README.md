# 🎸 Reconnaissance de son dans la musique - Projet Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Gradio](https://img.shields.io/badge/Interface-Gradio-ff69b4)
![Status](https://img.shields.io/badge/Status-Terminé-success)
![Score](https://img.shields.io/badge/Score%20Test-71%25-brightgreen)

## 📖 Description
Ce dépôt contient notre travail pour le projet de Machine Learning de l'ESIEA. L'objectif est de concevoir un modèle de Deep Learning capable d'identifier automatiquement la présence d'instruments de musique spécifiques dans un fichier audio. 

Nous travaillons sur une version simplifiée du dataset IRMAS. Le modèle doit classifier quatre instruments cibles : la guitare acoustique (gac), l'orgue (org), le piano (pia) et la voix humaine (voi).

**Objectif principal atteint :** Le modèle devait identifier correctement au moins un instrument pour 70% des morceaux de l'ensemble de test. Notre modèle final atteint une précision de **91.91%** en validation et de **71.32%** sur le jeu de test.

## ⚙️ Architecture et Choix Techniques

Notre pipeline respecte le cahier des charges suivant:
1. **Transformation des données :** Conversion des ondes sonores brutes (`.wav`) en spectrogrammes de Mel (représentation 2D). L'amplitude est passée en décibels pour une meilleure perception des harmoniques par le modèle. Ces transformations sont sauvegardées en tenseurs PyTorch (`.pt`) en amont de l'entraînement pour minimiser le temps de calcul.
2. **Architecture du Modèle :** Réseau de neurones convolutifs (CNN). Pour gérer les entrées de taille variable du jeu de test (morceaux de différentes longueurs), nous utilisons une couche de pooling adaptatif (`AdaptiveAvgPool2d`) juste avant les couches denses.
3. **Lutte contre le sur-apprentissage :** Implémentation de couches de `Dropout` et de `BatchNorm2d`. Division du jeu d'entraînement initial en 75% train / 25% validation.
4. **Évaluation :** Test sur des morceaux de longueurs variables avec une métrique binaire (1 si l'instrument cible est détecté, 0 sinon).

## 🚀 Installation et Prérequis

Ce projet utilise l'accélération GPU (CUDA) via PyTorch pour des temps d'apprentissage optimisés.

1. Clonez ce dépôt.
2. Assurez-vous d'avoir récupéré le dossier de données `Dataset` (non inclus dans le repo Git).
3. Installez les dépendances via le fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```
*(Dépendances principales : torch, librosa, numpy, tqdm, gradio).*

## 💻 Utilisation

Le projet est divisé en plusieurs scripts modulaires à exécuter dans l'ordre :

### 1. Pré-traitement
Convertit les fichiers audio d'entraînement en spectrogrammes sauvegardés sous `processed_data/train/`.
```bash
python data_preprocessing.py
```

### 2. Entraînement
Lance l'entraînement du modèle CNN, sauvegarde le meilleur modèle dans le dossier `models/` et affiche les métriques de validation.
```bash
python train.py
```

### 3. Évaluation sur l'ensemble de Test
Évalue le modèle entraîné sur l'ensemble de test (fichiers `.wav` de longueurs variables avec labels au format `.txt`) et calcule le score final.
```bash
python evaluate.py
```

### 4. 🎁 Bonus : Interface Web Interactive
Comme suggéré dans le cahier des charges (bonus exploitant le modèle), nous avons développé une interface graphique minimaliste avec Gradio. Elle permet de glisser-déposer n'importe quel fichier audio et d'obtenir les probabilités de présence pour chaque instrument en temps réel.
```bash
python app_bonus.py
```

## 👥 Auteurs et Remerciements
* **Matéo LETERTRE** - Étudiant en 4ème année de cycle ingénieur
* **Matthieu HAMON** - Étudiant en 4ème année de cycle ingénieur
* **Tom PANNIER** - Étudiant en 4ème année de cycle ingénieur
* **Nicolas BLANCHARD** - Étudiant en 4ème année de cycle ingénieur

**Mentions IA Générative :** Dans le cadre de ce projet, une IA générative (Google Gemini) a été utilisée à titre d'assistant technique. Elle a servi à structurer l'architecture des dossiers, à fournir des pistes d'implémentation pour le nettoyage des labels (expressions régulières / parsing des fichiers `.txt` de l'ensemble de test), et à générer la base du code de l'interface web (Gradio) pour le livrable bonus. Toute la logique mathématique et l'entraînement ont été vérifiés et exécutés localement par l'équipe.
```