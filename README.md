![](https://miro.medium.com/max/3200/1*xJ0MjCfbwQ4v2dhXe3JCIg.png)
# Moteur de recherche par image - Brique IA
Retrouver les images similaires dans une base de données pour faciliter la recherche par image

## Résumé
La recherche par image est l'une des applications de la vision par ordinateur. L'idée de travailler sur ce moteur date du bootcamp Africa Tech Up Tour de juillet 2019 à Lomé. Elom et moi, nous avions proposer de faire des recherches sur des sites touristiques africains (togolais en particulier).

J'ai utilisé Tensorflow pour ce bon de et la librairie ANNOY pour.

## Sommaire
1. Entraînement d'un modèle de classification
2. Compresser les images avec l'extracteur du modèle entraîné plus tôt
3. Contruction du graphe

## Outils
**Tensorflow (Google)**<br>
TensorFlow est un outil open source d'apprentissage automatique développé par Google. Le code source a été ouvert le 9 novembre 2015 par Google et publié sous licence Apache. Il est fondé sur l'infrastructure DistBelief, initiée par Google en 2011, et est doté d'une interface pour Python, Julia et R.

**Annoy (Spotify)**<br>
Annoy (Approximate Nearest Neighbor Oh Yeah), est une bibliothèque open-source pour l'implémentation approximative du plus proche voisin.

## I. Entraînement d'un modèle de classification
L'objectif est 
Pour gagner du temps et être plus efficace, j'ai opté pour le transfer learning

## II. Compression des images
Une fois le modèle convolutif entraîné, son extracteur est utilisé pour compresser les images

## III. Construction du graphe ANNOY
Une fois les images compressées, on peut les passer à ANNOY

Je vais l'utiliser pour trouver les vecteurs de caractéristiques d'image dans un ensemble donné qui est le plus proche (ou le plus similaire) d'un vecteur de caractéristiques donné.

Il n'y a que deux paramètres principaux nécessaires pour régler Annoy: le nombre d'arbres `n_trees` et le nombre de nœuds à inspecter lors de la recherche `search_k`.

`n_trees` est fourni pendant la construction et affecte le temps de construction et la taille de l'index. Une valeur plus élevée donnera des résultats plus précis, mais des index plus grands.

`search_k` est fourni lors de l'exécution et affecte les performances de recherche. Une valeur plus élevée donnera des résultats plus précis, mais prendra plus de temps pour revenir.


## Implémentation
Le code suivant est un exemple d'utilisation de ANNOY
```python
import annoy
import numpy as np

x = np.random.uniform(-12, 10, (10000, 20))

t = annoy.AnnoyIndex(x.shape[1], 'angular')

for i in range(x.shape[0]):
    t.add_item(i, x[i])

t.build(100)

t.save('graph.ann')
```

## Livrables
Les livrables sont le modèle (l'extracteur pour compresser les images des clients) et le graphe ANNOY qui permetra de chercher les images similaires.


## Références
* https://github.com/spotify/annoy
* https://towardsdatascience.com/image-similarity-detection-in-action-with-tensorflow-2-0-b8d9a78b2509
* https://www.slideshare.net/RobinAndreauReni/recommendation-systems-109609898

## Licence
Ce projet est sous licence MIT Licence

---
*Pour plus d’informations, merci de contacter joseph.kakone@gmail.com*
