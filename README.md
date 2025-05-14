# Análisis de Pseudociencias: Detección de Sentimientos en Tweets

Este proyecto implementa un análisis automatizado de tweets relacionados con pseudociencias, especialmente astrología, mediante técnicas de procesamiento de lenguaje natural y aprendizaje automático.

## Contenido de la Carpeta

- **analisis_pseudociencias.py**: Script principal que implementa el análisis de sentimientos, clustering y visualizaciones.
- **requirements.txt**: Lista de dependencias necesarias para ejecutar el proyecto.
- **Archivos de diccionario**:
  - **diccionario_Final_Astrologia.csv**: Diccionario con términos de astrología y sus frecuencias.
  - **AnalisisPositivo.csv**: Palabras clasificadas como positivas.
  - **AnalisisNegativo.csv**: Palabras clasificadas como negativas.
  - **AnalisisNeutral.csv**: Palabras clasificadas como neutrales.
- **Astrologia/**: Directorio con archivos CSV que contienen tweets sobre astrología.
- **graficas/**: Directorio donde se guardan las visualizaciones generadas.
- **resultados_analisis.csv**: Archivo con los resultados del análisis (generado por el script).

## Objetivo del Proyecto

El objetivo principal es analizar el sentimiento de tweets relacionados con pseudociencias y descubrir patrones en cómo las personas hablan sobre estos temas en redes sociales. El proyecto utiliza:

1. **Clasificación basada en diccionario**: Identifica palabras positivas, negativas y neutrales para determinar el sentimiento general de los tweets.
2. **Aprendizaje no supervisado (K-means)**: Agrupa tweets similares sin etiquetas previas.
3. **Aprendizaje supervisado (Naive Bayes)**: Clasifica tweets según su sentimiento, aprendiendo de ejemplos etiquetados.

## Funcionalidades Principales

### 1. Análisis de Sentimientos
- Clasifica automáticamente tweets como positivos, negativos o neutrales.
- Utiliza diccionarios de palabras precargados para identificar el sentimiento predominante.

### 2. Clustering con K-means
- Agrupa tweets similares en clusters basados en su contenido.
- Identifica las palabras más relevantes para cada cluster.
- Descubre tendencias y temas comunes en las conversaciones sobre pseudociencias.

### 3. Clasificación con Naive Bayes
- Entrena un modelo para predecir automáticamente el sentimiento de nuevos tweets.
- Evalúa la precisión del modelo mediante división en conjuntos de entrenamiento y prueba.

### 4. Visualizaciones
- **Gráficos de barras**: Distribución de tweets por sentimiento y cluster.
- **Gráficos circulares**: Proporción de sentimientos en los tweets analizados.
- **Nubes de palabras**: Visualiza las palabras más frecuentes por sentimiento y cluster.

## Resultados

El análisis revela:

- Distribución de sentimientos hacia la astrología (positivos, negativos y neutrales).
- Palabras clave asociadas con cada tipo de sentimiento.
- Agrupaciones naturales de tweets que revelan diferentes aspectos del discurso sobre pseudociencias.
- Un clasificador automático capaz de predecir el sentimiento de nuevos tweets.

## Cómo Ejecutar el Proyecto

1. **Instalar dependencias**:
   ```
   pip install -r requirements.txt
   ```

2. **Ejecutar el análisis**:
   ```
   python analisis_pseudociencias.py
   ```

3. **Revisar resultados**:
   - Examinar el archivo `resultados_analisis.csv` para ver los tweets clasificados.
   - Explorar el directorio `graficas/` para ver las visualizaciones generadas.

## Estructura del Código

El script principal (`analisis_pseudociencias.py`) está organizado en funciones modulares:

- **cargar_tweets()**: Lee los archivos CSV con tweets desde el directorio Astrologia/.
- **cargar_listas_sentimientos()**: Carga las listas de palabras positivas, negativas y neutrales.
- **limpiar_texto()**: Preprocesa el texto eliminando caracteres innecesarios.
- **clasificar_tweets()**: Asigna etiquetas de sentimiento según palabras clave.
- **aplicar_kmeans()**: Implementa clustering no supervisado.
- **aplicar_naive_bayes()**: Entrena y evalúa un clasificador supervisado.
- **generar_visualizaciones()**: Crea gráficos para facilitar el análisis.
- **main()**: Coordina todo el proceso de análisis.

## Extensiones Posibles

Este proyecto puede expandirse de varias maneras:

1. **Análisis temporal**: Estudiar cómo cambia el sentimiento hacia las pseudociencias a lo largo del tiempo.
2. **Comparación entre pseudociencias**: Comparar cómo se habla de diferentes pseudociencias.
3. **Análisis de redes**: Estudiar quién habla de pseudociencias y cómo se difunde la información.
4. **Modelos más avanzados**: Implementar algoritmos de deep learning para mejorar la clasificación.