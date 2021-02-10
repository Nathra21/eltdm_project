# eltdm_project

## Mise en place de l'environnement

Il est possible d'avoir un environnement Jupyter avec GPU en utilisant Docker.
Voir [ce repository](https://github.com/iot-salzburg/gpu-jupyter).

Pour installer les packages,
```bash
pip3 install -r requirements.txt
```

## Exécution des tests

Des tests sont à disposition pour vérifier la bonne exécution de l'implémentation sous PyTorch de la Random Forest.

```bash
cd eltdm_project
python3 -m pytest test/
```

## Auteurs

* Etienne BOISSEAU
* Olivier DULCY
