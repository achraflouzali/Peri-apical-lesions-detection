# Entraînement du modèle yolov5 sur les images radiographiques

## 1) Création d'un environnement et l'installation des packages nécessaires

Pour l'utilisation de ce modèle, il faut avoir une version de Python entre 3.7 et 3.10.

### a) Création et activation de l'environnement 

Sur une invite de commandes on exécutes les commandes suivantes pour créer et activer l'environnement (Je l'ai nommé Projet3A)
py -m pip install --upgrade pippy -m pip install --user virtualenvpy -m venv Projet3A.\Projet3A\Scripts\activatecd Projet3A
### b) Importation du modèle Yolov5 initial et installation des packages nécessaires

git clone https://github.com/ultralytics/yolov5cd yolov5git reset --hard 064365d8683fd002e9ad789c1e91fa3d021b44f0
Après qu'on a importé le modèle Yolov5, il faut installer les packages nécessaires par de suite, d'où sert le fichier requirements.txt dans le dossier yolov5. Or, on utilise d'autres packages en plus des packages du modèle initial et pour régler ceci j'ai modifié le fichier requirements.txt en y rajoutant les autres packages nécessaires

Donc il faut remplacer le fichier requirements.txt dans le fichier requirement que vous pouvez trouver sur le github avec le même nom (requirements.txt) et éxécuter la commande suivante
pip install -r requirements.txt
### c) Structure des données

Vu qu'on utilise le modèle initial pour l'adapter à notre dataset, il faut inclure les fichiers de notre dataset dans le dossier yolov5.
Premièrement, il faut mettre le fichier lesions.yaml (un fichier indiquant au modèle les chemins d'accès des bases de données d'entraînement, validation et test) dans le dossier data.

Deuxièmement, il faut mettre le script lesions_data.py (fichier de conversion et répartition des données) dans le dossier yolov5 .

Après qu'on a organisé le dossier yolov5, on peut passer à la conversion et répartition des donnnées en premiers temps: Pour cela, il faut avoir les chemins d'accès des deux dossiers (dossier pour les fichiers xml et un autre pour les images DICOM)
python lesions_f.py chemin/vers/fichier/xml  chemin/vers/images
Cette commande permet de:
1) Convertir les images DICOM en images png de dimensions (512,256) (modifiables en modifiant la fonction  convert_dicom_folder_to_png)

2) Convertir les fichiers xml en des fichier txt

3) Répartir la base de données en 80% train, 10% validation, 10% test (Pourcentages modifiables en modifiant les arguments de la fonction train_test_split)

### d) Entraînement

On lance l'entraînement en exécutant le fichier train.py avec les arguments suivant :

img: taille de l'image 

epochs: nombre d'epochs (itérations sur le dataset entier) 

data: chemin d'accés vers le fichier yaml du dataset

batch: taille du batch 

cfg: Configuration du modèle

weights: Pour montrer au modèle les poids donnés au paramétres au début
    

La configuration optimale a été trouvée à l'aide de cette configuration (Entraînement du modèle yolov5s sur des images de (1200,1200) avec un batch de 32)
python train.py --data lesions.yaml --epochs 600 --weights '' --img 1200 --cfg ./models/yolov5s.yaml
Après l'entraînement, le modèle crée un dossier train dans runs contenant les résultats d'entraînemnt, les paramètres du modèle optimal trouvé, et la prédiction du modèle sur les images de validation


```python

```
