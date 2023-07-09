# Reviews Movies

creacion de modelo para calificar una oracion
usando reseñas de peliculas

## Requerimientos

Se necesitan las siguientes dependencias de python 3.8

```bash
  pip install pandas
```

```bash
  pip install scikit-learn
```

```bash
  pip install "tensorflow<2.10"
```

```bash
  pip install matplotlib
```

```bash
  pip install gdown


```

## Desargar reseñas

```bash
mkdir data

cd data
```

```bash
gdown --id 1iN4CfzhuXFWzMaplAbrzlJPKoYgIW30c
```

## Entrenar modelo

```bash
python model.py
```

## Running Tests

Pruebas de prediccion del modelo

```bash
python predict.py
```

## Resultados del test 🚀

| Oracion                                         | Review      |
| ----------------------------------------------- | ----------- |
| Terrible servicio, no lo recomiendo             | NEGATIVA 😡 |
| Excelente servicio, muy recomendado             | POSITIVA 😄 |
| Una de las peliculas mas aburridas que he visto | NEGATIVA 😡 |
| La forma en que me trata es la mas horrible     | NEGATIVA 😡 |
| Amo la pelicula de principio a fin              | POSITIVA 😄 |
