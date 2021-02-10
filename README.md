# Projet ELTDM : Optimisation de l'inférence d'un classifieur RandomForest en utilisant une représentation matricielle

Notre étude s'intéresse à la réduction du temps de prédiction d'une RandomForest. Pour cela, nous nous sommes inspirés de l'article
[Taming Model Serving Complexity, Performance and Cost: A Compilation to Tensor Computations Approach](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf).

## Organisation du répertoire

```bash
.
├── random_forest_pytorch
│   ├── random_forest_gemm.py
│   └── utils.py
├── README.md
├── Report.ipynb
├── Report.pdf
├── requirements.txt
└── test
    └── test_RandomForestGEMM.py
```

Notre rapport est disponible au format Jupyter Notebook ainsi qu'en PDF.
L'implémentation principale de la RandomForest sous ``numpy`` et ``pytorch`` se trouve dans le fichier ``random_forest_gemm.py``.
Les fonctions auxiliaires utilisées pour le rapport et l'implémentation sont regroupées dans ``utils.py``.

Enfin, la liste des packages nécessaires à l'exécution de notre implémentation est disponible dans
``requirements.txt`` et des tests sont disponibles pour vérifier que les prédictions de notre implémentation
sont identiques à celles de la RandomForest.

### Mise en place de l'environnement

Pour installer les packages, il suffit de lancer dans un terminal la commande suivante :
```bash
pip3 install -r requirements.txt
```
Il est recommandé d'utiliser une version de pytorch ``>=1.7.0``.

Pour bénéficier d'un environnement clé en main, il est possible d'utiliser une image Jupyter sur Docker :
voir [ce repository](https://github.com/iot-salzburg/gpu-jupyter).

### Exécution des tests

L'implémentation de la RandomForest sous forme matricielle, disponible sous ``numpy`` ou ``pytorch``, est testée sur 2 jeux de données synthétiques.
Pour exécuter les tests, il suffit de se placer à la racine du dossier, puis d'utiliser la commande suivante :
```bash
python3 -m pytest test/
```

## Auteurs

* Etienne BOISSEAU
* Olivier DULCY
