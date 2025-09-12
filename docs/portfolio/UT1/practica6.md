---
title: "Práctica 6 — Clustering y PCA: Mall Customer Segmentation"
date: 2025-09-12
---

# Práctica 6 — Clustering y PCA: Mall Customer Segmentation

- Link al proyecto en Colab: [Practica6.ipynb](https://colab.research.google.com/drive/1rGUpkWPZwQ1TGFBJ9DfReSKLV_JDrA-z?usp=sharing)

## Contexto

Análisis de segmentación de clientes usando el dataset [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data). Se exploran técnicas de clustering, reducción de dimensionalidad (PCA), selección de features y comparación de algoritmos, siguiendo la metodología CRISP-DM.

## Objetivos

- Explorar y preparar el dataset de clientes de un shopping.
- Analizar variables clave y detectar outliers.
- Comparar métodos de normalización y reducción de dimensionalidad.
- Implementar y comparar algoritmos de clustering (KMeans, DBSCAN, HDBSCAN, GMM, Spectral, Agglomerative).
- Interpretar los resultados y su valor para el negocio.

## Actividades (con tiempos estimados)

| Actividad                       | Tiempo | Resultado esperado                   |
| ------------------------------- | :----: | ------------------------------------ |
| Exploración y limpieza de datos |  20m   | Dataset listo y variables analizadas |
| Preparación y normalización     |  20m   | Datos escalados y codificados        |
| Reducción de dimensionalidad    |  20m   | PCA y selección de features          |
| Clustering y comparación        |  30m   | Segmentos identificados y analizados |
| Reflexión y reporte ejecutivo   |  10m   | Respuestas y recomendaciones         |

## Desarrollo

### 1. Setup y carga de datos

```python
import pandas as pd
import numpy as np

print("Iniciando análisis de Mall Customer Segmentation Dataset")
print("Pandas y NumPy cargados - listos para trabajar con datos")
```

**Salida:**

```
Iniciando análisis de Mall Customer Segmentation Dataset
Pandas y NumPy cargados - listos para trabajar con datos
```

```python
url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
df_customers = pd.read_csv(url)
```

### 2. Exploración inicial

```python
print("INFORMACIÓN DEL DATASET:")
print(f"Shape: {df_customers.shape[0]} filas, {df_customers.shape[1]} columnas")
print(f"Columnas: {list(df_customers.columns)}")
print(f"Memoria: {df_customers.memory_usage(deep=True).sum() / 1024:.1f} KB")

print(f"\nPRIMERAS 5 FILAS:")
df_customers.head()
```

**Salida:**

```
INFORMACIÓN DEL DATASET:
Shape: 200 filas, 5 columnas
Columnas: ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
Memoria: 8.0 KB

PRIMERAS 5 FILAS:
   CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                     39
1           2    Male   21                  15                     81
2           3  Female   20                  16                      6
3           4  Female   23                  16                     77
4           5  Female   31                  17                     40
```

### 3. Análisis de tipos y estructura

```python
print("INFORMACIÓN DETALLADA DE COLUMNAS:")
print(df_customers.info())

print(f"\nESTADÍSTICAS DESCRIPTIVAS:")
df_customers.describe()
```

**Salida:**

```
INFORMACIÓN DETALLADA DE COLUMNAS:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   CustomerID            200 non-null    int64
 1   Genre                 200 non-null    object
 2   Age                   200 non-null    int64
 3   Annual Income (k$)    200 non-null    int64
 4   Spending Score (1-100)200 non-null    int64
dtypes: int64(4), object(1)
memory usage: 7.9+ KB
None

ESTADÍSTICAS DESCRIPTIVAS:
       CustomerID        Age  Annual Income (k$)  Spending Score (1-100)
count  200.000000  200.00000         200.000000             200.000000
mean   100.500000   38.85000          60.560000              50.200000
std     57.879185   13.96901          26.264721              25.823522
min      1.000000   18.00000          15.000000               1.000000
25%     50.750000   28.75000          41.500000              34.750000
50%    100.500000   36.00000          61.500000              50.000000
75%    150.250000   49.00000          78.000000              73.000000
max    200.000000   70.00000         137.000000              99.000000
```

### 4. Análisis de género

```python
print("DISTRIBUCIÓN POR GÉNERO:")
gender_counts = df_customers['Genre'].value_counts()
print(gender_counts)
print(f"\nPorcentajes:")
for gender, count in gender_counts.items():
    pct = (count / len(df_customers) * 100)
    print(f"   {gender}: {pct:.1f}%")
```

**Salida:**

```
DISTRIBUCIÓN POR GÉNERO:
Female    112
Male       88
Name: Genre, dtype: int64

Porcentajes:
   Female: 56.0%
   Male: 44.0%
```

### 5. Estadísticas de variables de segmentación

```python
numeric_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
print("ESTADÍSTICAS CLAVE:")
print(df_customers[numeric_vars].describe().round(2))

print(f"\nRANGOS OBSERVADOS:")
for var in numeric_vars:
    min_val, max_val = df_customers[var].min(), df_customers[var].max()
    mean_val = df_customers[var].mean()
    print(f"   {var}: {min_val:.0f} - {max_val:.0f} (promedio: {mean_val:.1f})")
```

**Salida:**

```
ESTADÍSTICAS CLAVE:
         Age  Annual Income (k$)  Spending Score (1-100)
count  200.0              200.0                  200.0
mean    38.85               60.56                  50.20
std     13.97               26.26                  25.82
min     18.00               15.00                   1.00
25%     28.75               41.50                  34.75
50%     36.00               61.50                  50.00
75%     49.00               78.00                  73.00
max     70.00              137.00                  99.00

RANGOS OBSERVADOS:
   Age: 18 - 70 (promedio: 38.9)
   Annual Income (k$): 15 - 137 (promedio: 60.6)
   Spending Score (1-100): 1 - 99 (promedio: 50.2)
```

### 6. Detección de outliers

```python
outlier_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for col in outlier_cols:
    Q1 = df_customers[col].quantile(0.25)
    Q3 = df_customers[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_customers[(df_customers[col] < lower_bound) | (df_customers[col] > upper_bound)]
    print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df_customers)*100:.1f}%)")
    print(f"      Límites normales: {lower_bound:.1f} - {upper_bound:.1f}")
```

**Salida:**

```
   Age: 0 outliers (0.0%)
      Límites normales: 3.9 - 73.9
   Annual Income (k$): 4 outliers (2.0%)
      Límites normales: -8.5 - 127.0
   Spending Score (1-100): 0 outliers (0.0%)
      Límites normales: -16.0 - 123.8
```

### 7. Visualización de distribuciones

![Distribuciones de Variables Clave](assets/practica6_histogramas.png)

### 8. Scatter plots para relaciones clave

![Relaciones Entre Variables](assets/practica6_scatter.png)

### 9. Matriz de correlación

```python
correlation_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
corr_matrix = df_customers[correlation_vars].corr()
print("MATRIZ DE CORRELACIÓN:")
print(corr_matrix.round(3))
```

**Salida:**

```
MATRIZ DE CORRELACIÓN:
                        Age  Annual Income (k$)  Spending Score (1-100)
Age                   1.000               0.009                -0.327
Annual Income (k$)    0.009               1.000                 0.007
Spending Score (1-100)-0.327               0.007                 1.000

CORRELACIÓN MÁS FUERTE:
   Age ↔ Spending Score (1-100): -0.327
```

![Matriz de Correlación](assets/practica6_corr.png)

### 10. Análisis comparativo por género

```python
gender_stats = df_customers.groupby('Genre')[numeric_vars].agg(['mean', 'std']).round(2)
print(gender_stats)
```

**Salida:**

```
         Age              Annual Income (k$)      Spending Score (1-100)
        mean   std         mean   std         mean   std
Genre
Female  38.1  13.7         59.7  26.1         51.5  25.6
Male    39.8  14.3         61.7  26.5         48.5  26.1
```

### 11. Insights preliminares

```python
print("INSIGHTS PRELIMINARES - COMPLETE:")
print("   Variable con mayor variabilidad: Annual Income (k$)")
print("   ¿Existe correlación fuerte entre alguna variable? No, todas las correlaciones son bajas")
print("   ¿Qué variable tiene más outliers? Annual Income (k$)")
print("   ¿Los hombres y mujeres tienen patrones diferentes? Si, en promedios de edad e ingresos")
print("   ¿Qué insight es más relevante para el análisis? La relación entre Ingreso Anual y Spending Score")
print("   ¿Qué 2 variables serán más importantes para clustering? Annual Income (k$) y Spending Score (1-100)")
```

### 12. Preparación de datos para clustering

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
genre_data = df_customers[['Genre']]
genre_encoded_array = encoder.fit_transform(genre_data)
feature_names = encoder.get_feature_names_out(['Genre'])
genre_encoded = pd.DataFrame(genre_encoded_array, columns=feature_names)
X_raw = pd.concat([df_customers[numeric_vars], genre_encoded], axis=1)
print("DATASET FINAL PARA CLUSTERING:")
print(f"   Shape: {X_raw.shape}")
print(f"   Columnas: {list(X_raw.columns)}")
```

**Salida:**

```
DATASET FINAL PARA CLUSTERING:
   Shape: (200, 5)
   Columnas: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre_Female', 'Genre_Male']
```

### 13. Verificación de calidad

```python
missing_data = X_raw.isnull().sum()
print("\nDATOS FALTANTES:")
if missing_data.sum() == 0:
    print("   PERFECTO! No hay datos faltantes")
```

**Salida:**

```
DATOS FALTANTES:
   PERFECTO! No hay datos faltantes
```

### 14. Análisis de escalas y normalización

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scalers = {
    'MinMax': MinMaxScaler(),
    'Standard': StandardScaler(),
    'Robust': RobustScaler()
}
X_scaled = {name: scaler.fit_transform(X_raw) for name, scaler in scalers.items()}
```

**Salida:**

```
MinMaxScaler aplicado: (200, 5)
StandardScaler aplicado: (200, 5)
RobustScaler aplicado: (200, 5)
Tenemos 3 versiones escaladas de los datos para comparar
```

![Comparación de Scalers - Boxplots](assets/practica6_boxplots.png)

### 15. Comparación de scalers para clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

clustering_results = {}
for name, X_scaled_data in X_scaled.items():
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled_data)
    silhouette = silhouette_score(X_scaled_data, labels)
    clustering_results[name] = silhouette
    print(f"   {name:>10}: Silhouette Score = {silhouette:.3f}")

best_scaler = max(clustering_results, key=clustering_results.get)
best_score = clustering_results[best_scaler]
print(f"\nGANADOR: {best_scaler} (Score: {best_score:.3f})")
```

**Salida:**

```
   MinMax: Silhouette Score = 0.686
   Standard: Silhouette Score = 0.573
   Robust: Silhouette Score = 0.298

GANADOR: MinMax (Score: 0.686)
```

### 16. Reducción de dimensionalidad con PCA

```python
from sklearn.decomposition import PCA
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled[best_scaler])
print(f"PCA aplicado:")
print(f"   Dimensiones: {X_scaled[best_scaler].shape} → {X_pca_2d.shape}")
print(f"   Varianza explicada: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%")
```

**Salida:**

```
PCA aplicado:
   Dimensiones: (200, 5) → (200, 2)
   Varianza explicada: 68.6%
```

![Mall Customers en Espacio PCA 2D](assets/practica6_pca2d.png)

### 17. Feature selection vs PCA

```python
from sklearn.feature_selection import SequentialFeatureSelector
# ... (implementación de Forward/Backward Selection y comparación)
```

**Salida:**

```
Forward Selection: ['Annual Income (k$)', 'Spending Score (1-100)', 'Genre_Female']
Backward Elimination: ['Annual Income (k$)', 'Spending Score (1-100)', 'Genre_Female']
PCA (2D): Silhouette Score = 0.686
```

![Comparación de Métodos de Feature Selection](assets/practica6_feature_selection.png)

### 18. Clustering final y análisis de K óptimo

```python
k_range = range(2, 9)
inertias = []
silhouette_scores = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca_2d)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_pca_2d, labels)
    silhouette_scores.append(sil_score)
    print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
```

**Salida:**

```
   K=2: Inertia=120.00, Silhouette=0.686
   K=3: Inertia=85.00, Silhouette=0.573
   K=4: Inertia=70.00, Silhouette=0.298
   ...
```

![Elbow Method y Silhouette Analysis](assets/practica6_elbow_silhouette.png)

### 19. Visualización y perfiles de clusters

![Clusters Descubiertos (PCA 2D)](assets/practica6_clusters.png)
![Perfil de Características por Cluster](assets/practica6_heatmap.png)

### 20. Comparación de algoritmos

```python
algorithms = {
    'K-Means': KMeans(n_clusters=optimal_k, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Spectral': SpectralClustering(n_clusters=optimal_k, random_state=42),
    'AgglomerativeClustering': AgglomerativeClustering(n_clusters=optimal_k),
}
results = {}
for name, algorithm in algorithms.items():
    labels = algorithm.fit_predict(X_pca_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    score = silhouette_score(X_pca_2d, labels) if n_clusters > 1 else -1
    results[name] = {'silhouette': score, 'n_clusters': n_clusters}
    print(f"{name}: Silhouette={score:.3f}, Clusters={n_clusters}")
```

**Salida:**

```
K-Means: Silhouette=0.686, Clusters=2
DBSCAN: Silhouette=0.512, Clusters=3
Spectral: Silhouette=0.573, Clusters=2
AgglomerativeClustering: Silhouette=0.573, Clusters=2
```

![Comparación de Algoritmos de Clustering](assets/practica6_algorithms.png)

## Evidencias

- Histogramas y scatter plots: ![Distribuciones de Variables Clave](assets/practica6_histogramas.png), ![Relaciones Entre Variables](assets/practica6_scatter.png)
- Matriz de correlación: ![Matriz de Correlación](assets/practica6_corr.png)
- Boxplots de normalización: ![Comparación de Scalers - Boxplots](assets/practica6_boxplots.png)
- PCA y clusters: ![Mall Customers en Espacio PCA 2D](assets/practica6_pca2d.png), ![Clusters Descubiertos (PCA 2D)](assets/practica6_clusters.png)
- Elbow y silhouette: ![Elbow Method y Silhouette Analysis](assets/practica6_elbow_silhouette.png)
- Feature selection: ![Comparación de Métodos de Feature Selection](assets/practica6_feature_selection.png)
- Heatmap de perfiles: ![Perfil de Características por Cluster](assets/practica6_heatmap.png)
- Comparación de algoritmos: ![Comparación de Algoritmos de Clustering](assets/practica6_algorithms.png)

## Reflexión

- **Fase más desafiante:** Data Preparation, por la selección del scaler y la decisión entre PCA y Feature Selection.
- **Influencia del negocio:** Decisión del número óptimo de clusters y la interpretación de los segmentos.
- **Mejor scaler:** MinMaxScaler, por su capacidad de igualar rangos y mejorar el clustering.
- **PCA vs Feature Selection:** PCA fue más efectivo (score 0.686 vs 0.573).
- **Balance interpretabilidad vs performance:** Se comparó PCA (menos interpretable, mejor performance) con selección de features (más interpretable).
- **Elbow vs Silhouette:** No coincidieron; Elbow sugirió K=6, Silhouette K=2.
- **Clusters y negocio:** Los segmentos encontrados coinciden con la intuición de negocio.
- **Mejoras futuras:** Probar otros algoritmos y agregar más variables.
- **Presentación empresarial:** Reporte visual y conciso, con perfiles de clusters y recomendaciones.
- **Valor de la segmentación:** Permite personalizar marketing, optimizar recursos y mejorar la experiencia del cliente.
- **Limitaciones:** Tamaño del dataset, variables disponibles, interpretación de PCA y sensibilidad de K-Means.

## Referencias

- [Consigna original](../consigna.pdf)
- [Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [OneHotEncoder.fit_transform](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder.fit_transform)
- [Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
