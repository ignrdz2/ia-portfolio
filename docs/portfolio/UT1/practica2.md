---
title: "Práctica 2 — Feature Engineering simple + Modelo base"
date: 2025-01-01
---

# Práctica 2 — Feature Engineering simple + Modelo base

## Contexto

En esta práctica se explora el preprocesamiento de datos y la creación de nuevas variables (feature engineering) para el dataset del Titanic. Se implementa un modelo base de Regresión Logística y se compara con un baseline usando DummyClassifier.

## Objetivos

- Investigar componentes clave de Scikit-learn.
- Realizar imputación de valores faltantes y crear nuevas features.
- Entrenar y comparar modelos base y baseline.
- Analizar métricas y errores del modelo.

## Actividades (con tiempos estimados)

| Actividad                      | Tiempo | Resultado esperado                  |
| ------------------------------ | :----: | ----------------------------------- |
| Investigación de Scikit-learn  |  10m   | Resumen de componentes y parámetros |
| Feature engineering y limpieza |  15m   | Nuevas columnas y datos listos      |
| Modelo base y baseline         |  20m   | Métricas y comparación              |

## Desarrollo

### 0. Investigación de Scikit-learn

- **LogisticRegression:** Resuelve problemas de clasificación binaria. Parámetros importantes: `solver`, `max_iter`, `penalty`, `random_state`. Usar `solver='liblinear'` para datasets pequeños y/o con regularización L1.
- **DummyClassifier:** Sirve para establecer un baseline, usando estrategias como `most_frequent`, `stratified`, etc. Es importante para saber si el modelo real supera a una predicción trivial.
- **train_test_split:** El parámetro `stratify` asegura que la proporción de clases se mantenga en train y test. `random_state` permite reproducibilidad. El porcentaje de test recomendado suele ser 20-30%.
- **Métricas:** `classification_report` muestra precisión, recall y f1-score por clase. La matriz de confusión permite ver los tipos de errores. `accuracy` mide aciertos globales, pero puede ser engañosa si las clases están desbalanceadas.

### 1. Preprocesamiento y Feature Engineering

```python
from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/IA-UT1')
except Exception:
    ROOT = Path.cwd() / 'IA-UT1'

DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
for d in (DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print('Outputs →', ROOT)

import pandas as pd
train = pd.read_csv(DATA_DIR / 'train.csv')
df = train.copy()

# Imputación de valores faltantes
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('median'))

# Nuevas features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(',\s*([^\.]+)\.')
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# Preparar datos para el modelo
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','SibSp','Parch']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

X.shape, y.shape
```

**Salida:**

```text
((891, 14), (891,))
```

### 2. Modelo base y baseline

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
baseline_pred = dummy.predict(X_test)

lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print('Baseline acc:', accuracy_score(y_test, baseline_pred))
print('LogReg acc  :', accuracy_score(y_test, pred))

print('\nClassification report (LogReg):')
print(classification_report(y_test, pred))

print('\nConfusion matrix (LogReg):')
print(confusion_matrix(y_test, pred))
```

**Salida:**

```text
Baseline acc: 0.6145251396648045
LogReg acc  : 0.8156424581005587

Classification report (LogReg):
              precision    recall  f1-score   support

           0       0.82      0.89      0.86       110
           1       0.80      0.70      0.74        69

    accuracy                           0.82       179
   macro avg       0.81      0.79      0.80       179
weighted avg       0.81      0.82      0.81       179

Confusion matrix (LogReg):
[[98 12]
 [21 48]]
```

## Evidencias

- Resultados de métricas y matriz de confusión.
- Nuevas columnas creadas: `FamilySize`, `IsAlone`, `Title`.

## Reflexión

La Regresión Logística supera ampliamente al baseline, mostrando que el modelo aprende patrones relevantes. El modelo acierta más en los casos de no supervivencia (clase 0), pero también logra buena precisión en la clase 1. El error más grave sería predecir que alguien sobrevivió cuando no lo hizo, por el contexto del problema. Una posible mejora sería crear una feature que combine edad y clase, o analizar el impacto de los títulos en el nombre con mayor detalle.

## Referencias

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Descripción del dataset Titanic en Kaggle](https://www.kaggle.com/competitions/titanic/data)
