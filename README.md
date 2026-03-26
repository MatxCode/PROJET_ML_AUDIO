# Reconnaissance de son dans la musique - Projet Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Gradio](https://img.shields.io/badge/Interface-Gradio-ff69b4)
![Status](https://img.shields.io/badge/Status-Terminé-success)
![Score](https://img.shields.io/badge/Score%20Test-71%25-brightgreen)

# Reconnaissance de sons musicaux – Projet Machine Learning

**Le projet :**  
Dans ce projet, on devait créer un modèle capable de reconnaître des instruments de musique dans des fichiers audio. On a travaillé sur une version simplifiée du dataset IRMAS, avec quatre instruments à détecter : guitare acoustique, orgue, piano et voix.

L’objectif était assez clair : réussir à identifier correctement au moins un instrument dans 70 % des morceaux du jeu de test. Au final, notre modèle atteint un peu plus de 71 % sur le test, donc on a validé l’objectif. Sur la validation, on monte même à presque 92 %.

**Comment on s’y est pris :**  
Au lieu de travailler directement sur le son brut, on a transformé les fichiers audio en spectrogrammes de Mel. Concrètement, ça revient à transformer le son en image (avec le temps et la fréquence), ce qui est beaucoup plus simple à exploiter avec un réseau de neurones.

On a ensuite utilisé un CNN classique. Le principal problème était que les fichiers audio n'ont pas tous la même durée, donc nous avons utilisé une couche de pooling adaptatif pour que le modèle puisse quand même fonctionner avec des tailles différentes.

Pour éviter que le modèle apprenne “par cœur”, on a ajouté du Dropout et de la normalisation (BatchNorm), et on a bien séparé nos données entre entraînement et validation.

**Installation :**  
Pour l'installer, il faut cloner le projet, récupérer le dataset (qui n’est pas dans le repo), puis installer les dépendances avec le fichier requirements.txt.

**Comment lancer le projet :**  
Le projet se fait en plusieurs étapes :  
On commence par transformer les fichiers audio, puis on entraîne le modèle, ensuite on l’évalue sur le jeu de test.

On a aussi fait un bonus : une interface avec Gradio. Ça permet de tester le modèle facilement en déposant un fichier audio et en voyant directement ce qu’il détecte.

**L’équipe :**  
Matéo Letertre  
Matthieu Hamon  
Tom Pannier  
Nicolas Blanchard  

**IA générative :**  
Dans le cadre de ce projet, nous nous sommes aidés d'une IA générative (Google Gemini) en tant qu'assistant technique. 

Nous l'avons ainsi utilisée pour l'architecture du projet, c'est-à-dire au niveau de la structure des dossiers et la séparation des différents fichiers (data_preprocessing.py, dataset_loader.py, model.py, train.py et evaluate.py).

Nous l'avons également utilisé pour la rédaction du code, ainsi, pour chaque fonction, nous avons donné nos attentes et les contraintes technique du sujet (par exemple l'utilisation de AdaptiveAvgPool2d pour gérer les tailles variables et l'implémentation de librosa pour convertir efficacement les ondes audio en spectrogrammes de Mel à l'échelle logarithmique avant l'entraînement).

Toute la partie logique, la compréhension du pooling adaptatif et les phases d'entraînement ont été vérifiées et validées de notre côté.
