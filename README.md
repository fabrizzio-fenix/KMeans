# Análisis de Tweets sobre Astrología

## Descripción
Este proyecto realiza un análisis de sentimiento y clustering de tweets relacionados con astrología, utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático. El análisis incluye:

1. Clasificación por sentimiento (positivo, negativo, neutral)
2. Agrupación de tweets similares mediante K-means
3. Visualizaciones para el análisis de resultados

## Características principales
- **Análisis de sentimiento**: Clasifica los tweets como positivos, negativos o neutrales basado en diccionarios específicos para astrología.
- **Clustering con K-means**: Agrupa tweets similares para identificar temas comunes.
- **Clasificación Naive Bayes**: Implementa un clasificador para predecir el sentimiento de nuevos tweets.
- **Visualizaciones**: Genera gráficos de barras, gráficos circulares y nubes de palabras para facilitar el análisis.
- **Procesamiento de diccionarios**: Limpia y procesa las frecuencias de palabras para un análisis más preciso.

## Estructura del proyecto
```
.
├── analisis_pseudociencias.py   # Script principal de análisis
├── Astrologia/                  # Carpeta con archivos CSV de tweets
│   ├── Astrologia1.csv
│   ├── Astrologia2.csv
│   └── ...
├── AnalisisPositivo.csv         # Diccionario original de palabras positivas
├── AnalisisNegativo.csv         # Diccionario original de palabras negativas
├── AnalisisNeutral.csv          # Diccionario original de palabras neutrales
├── diccionario_Final_Astrologia.csv # Diccionario general de astrología
├── astrologia_palabras_positivas.csv  # Palabras positivas de astrología procesadas
├── astrologia_palabras_negativas.csv  # Palabras negativas de astrología procesadas
├── astrologia_palabras_neutrales.csv  # Palabras neutrales de astrología procesadas
├── top_palabras_astrologia.csv  # Top 100 palabras más frecuentes de astrología
├── graficas/                    # Directorio donde se guardan las visualizaciones
├── resultados_analisis.csv      # Resultados del análisis
└── README.md                    # Documentación del proyecto
```

## Requisitos
- Python 3.6+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- WordCloud
- Seaborn

Puedes instalar todas las dependencias con:
```
pip install -r requirements.txt
```

## Uso
Para ejecutar el análisis completo:
```
python analisis_pseudociencias.py
```

## Modificaciones recientes
- **Filtrado por astrología**: El sistema está configurado para usar exclusivamente palabras relacionadas con astrología de los archivos de análisis de sentimiento.
- **Procesamiento mejorado de diccionarios**: Se han implementado funciones para limpiar y normalizar las frecuencias de palabras en todos los diccionarios.
- **Archivos adicionales**: El script ahora genera archivos CSV procesados con las palabras de astrología filtradas:
  - `astrologia_palabras_positivas.csv`: Palabras positivas con sus frecuencias normalizadas
  - `astrologia_palabras_negativas.csv`: Palabras negativas con sus frecuencias normalizadas
  - `astrologia_palabras_neutrales.csv`: Palabras neutrales con sus frecuencias normalizadas
  - `top_palabras_astrologia.csv`: Las 100 palabras más frecuentes del diccionario general
- **Procesamiento de tweets**: Se ha mejorado la limpieza de texto para eliminar URLs, menciones, hashtags y caracteres especiales.

## Resultados
Los resultados del análisis se guardan en:

1. **resultados_analisis.csv**: Contiene todos los tweets analizados con sus clasificaciones de sentimiento y cluster.
2. **graficas/**: Directorio con visualizaciones generadas:
   - **distribucion_sentimientos.png**: Gráfico de barras de la distribución de sentimientos.
   - **proporcion_sentimientos.png**: Gráfico circular de la proporción de sentimientos.
   - **nube_palabras_[sentimiento].png**: Nubes de palabras para cada tipo de sentimiento.
   - **distribucion_clusters.png**: Distribución de tweets por cluster.
   - **nube_palabras_cluster_[#].png**: Nubes de palabras para cada cluster.

## Modelo de Machine Learning
El proyecto implementa dos modelos de machine learning:

1. **K-means (no supervisado)**: Agrupa tweets similares en clusters basándose en su contenido. Utiliza vectorización TF-IDF para representar los textos numéricamente.

2. **Naive Bayes (supervisado)**: Clasificador para predecir el sentimiento de tweets. Utiliza un enfoque de bolsa de palabras (CountVectorizer) para la representación de texto.

## Interpretación de resultados
- Los tweets se clasifican mayoritariamente como neutrales (43.684), con pocos tweets positivos (618) y muy pocos negativos (4).
- Los clusters identifican diferentes temas dentro de los tweets de astrología:
  - Cluster 0 (6.426 tweets): Tweets personales y horóscopos diarios
  - Cluster 1 (2.159 tweets): Discusiones sobre creencias y el universo
  - Cluster 2 (35.721 tweets): Retweets y contenido informativo sobre astrología

## Análisis de palabras
El procesamiento mejorado de los diccionarios ha revelado que:
- Las palabras positivas más frecuentes para astrología son: "much" (585), "many" (517), "real" (171)
- Las palabras negativas más frecuentes son: "other" (428), "little bit" (417), "any less" (353)
- Las palabras más frecuentes del diccionario general son términos como "astrology" (15.195), "#Astrology" (12.846) y "for" (11.570)

## Autores
Equipo de análisis de pseudociencias