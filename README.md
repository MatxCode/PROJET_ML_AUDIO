# Reconnaissance de sons musicaux – Projet Machine Learning

**Le projet :**  
Dans ce projet, on devait créer un modèle capable de reconnaître des instruments de musique dans des fichiers audio. On a travaillé sur une version simplifiée du dataset IRMAS, avec quatre instruments à détecter : guitare acoustique, orgue, piano et voix.

L’objectif était assez clair : réussir à identifier correctement au moins un instrument dans 70 % des morceaux du jeu de test. Au final, notre modèle atteint un peu plus de 71 % sur le test, donc on a validé l’objectif. Sur la validation, on monte même à presque 92 %.

**Comment on s’y est pris :**  
Au lieu de travailler directement sur le son brut, on a transformé les fichiers audio en spectrogrammes de Mel. En gros, ça revient à transformer le son en image (avec le temps et la fréquence), ce qui est beaucoup plus simple à exploiter avec un réseau de neurones.

On a ensuite utilisé un CNN classique. Le petit défi, c’était que les fichiers audio n’ont pas tous la même durée, donc on a utilisé une couche de pooling adaptatif pour que le modèle puisse quand même fonctionner avec des tailles différentes.

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
On s’est aidés d’une IA (Google Gemini) surtout pour des aspects techniques : organisation du projet, manipulation des fichiers, et base de l’interface web.

Mais tout ce qui concerne le modèle, les choix techniques et les résultats, on l’a fait nous-mêmes et on a tout vérifié.