#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Análisis de Tweets relacionados con Pseudociencias

Este script realiza un análisis completo de tweets relacionados con pseudociencias,
específicamente para la astrología, aplicando técnicas de procesamiento de lenguaje
natural y aprendizaje automático para:
1. Clasificar tweets según su sentimiento (positivo, negativo, neutral)
2. Agrupar tweets similares mediante clustering con K-means
3. Predecir el sentimiento de tweets mediante clasificación con Naive Bayes
4. Generar visualizaciones para facilitar el análisis de los resultados

Autores: Equipo de análisis de pseudociencias
Fecha: 2024
"""

# Importación de librerías necesarias
import os  # Para manipulación de directorios y archivos
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np  # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualizaciones
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Para convertir texto en vectores numéricos
from sklearn.cluster import KMeans  # Para clustering no supervisado
from sklearn.naive_bayes import MultinomialNB  # Para clasificación supervisada
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento y prueba
from sklearn.metrics import accuracy_score, classification_report  # Para evaluar rendimiento del clasificador
from wordcloud import WordCloud  # Para generar nubes de palabras
import re  # Para expresiones regulares
import seaborn as sns  # Para visualizaciones avanzadas
from collections import Counter  # Para contar ocurrencias

# Configurar opciones de visualización para gráficos en español
import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['font.family'] = 'sans-serif'

# Obtener directorio base donde se encuentra el script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Definir rutas absolutas para archivos de datos
ASTROLOGIA_DIR = os.path.join(BASE_DIR, 'Astrologia')
DICCIONARIO_FILE = os.path.join(BASE_DIR, 'diccionario_Final_Astrologia.csv')

def cargar_tweets():
    """
    Carga todos los tweets de astrología disponibles en archivos CSV.

    Esta función:
    1. Busca todos los archivos CSV en el directorio de Astrología
    2. Carga cada archivo y extrae las columnas de ID y texto
    3. Filtra los tweets en español si la información de idioma está disponible
    4. Combina todos los tweets en un solo DataFrame

    Returns:
        pandas.DataFrame: DataFrame con los tweets cargados (columnas: 'Tweet Id', 'Text')
    """
    print("Cargando tweets...")
    print(f"Buscando archivos en: {ASTROLOGIA_DIR}")

    all_tweets = []

    # Verificar si el directorio existe
    if not os.path.exists(ASTROLOGIA_DIR):
        print(f"ERROR: El directorio {ASTROLOGIA_DIR} no existe.")
        return pd.DataFrame(columns=['Tweet Id', 'Text'])

    # Obtener lista de archivos CSV en el directorio
    csv_files = [f for f in os.listdir(ASTROLOGIA_DIR) if f.endswith('.csv')]

    if not csv_files:
        print("No se encontraron archivos CSV en el directorio.")

    # Procesar cada archivo CSV
    for csv_file in csv_files:
        file_path = os.path.join(ASTROLOGIA_DIR, csv_file)
        try:
            # Cargar el CSV con encoding adecuado
            df = pd.read_csv(file_path, encoding='latin1')

            # Verificar si existe la columna Text para extraer contenido
            if 'Text' in df.columns:
                # Filtrar solo tweets en español si está disponible la columna de idioma
                if 'Language' in df.columns:
                    spanish_tweets = df[df['Language'] == 'es']
                    all_tweets.append(spanish_tweets[['Tweet Id', 'Text']])
                else:
                    all_tweets.append(df[['Tweet Id', 'Text']])

        except Exception as e:
            print(f"Error al cargar {csv_file}: {e}")

    # Combinar todos los tweets en un solo DataFrame
    if all_tweets:
        combined_tweets = pd.concat(all_tweets)
        print(f"Total de tweets cargados: {len(combined_tweets)}")
        return combined_tweets
    else:
        print("No se pudieron cargar tweets.")
        return pd.DataFrame(columns=['Tweet Id', 'Text'])

def cargar_diccionario():
    """
    Carga el diccionario de palabras relacionadas con astrología y sus frecuencias.
    Este diccionario se utiliza para enriquecer el análisis de palabras relevantes.

    Returns:
        pandas.DataFrame: DataFrame con el diccionario cargado y procesado
    """
    print("Cargando diccionario de astrología...")

    try:
        # Cargar el diccionario desde el archivo CSV
        diccionario = pd.read_csv(DICCIONARIO_FILE, encoding='latin1')

        # Renombrar las columnas para mayor claridad
        if len(diccionario.columns) >= 2:
            diccionario.columns = ['Palabra', 'Frecuencia']

        # Limpiar frecuencias (convertir a valores numéricos)
        def limpiar_frecuencia(valor):
            if pd.isna(valor) or valor == '':
                return 0
            # Si es un número, convertirlo directamente
            if isinstance(valor, (int, float)):
                return float(valor)
            # Si es string, limpiar y convertir
            valor_str = str(valor).replace('"', '').replace(',', '')
            # Intentar convertir a float
            try:
                return float(valor_str)
            except ValueError:
                return 0

        diccionario['Frecuencia_Limpia'] = diccionario['Frecuencia'].apply(limpiar_frecuencia)

        # Ordenar por frecuencia descendente y eliminar valores con frecuencia 0
        diccionario = diccionario[diccionario['Frecuencia_Limpia'] > 0].sort_values('Frecuencia_Limpia', ascending=False)

        print(f"Diccionario cargado con {len(diccionario)} términos de astrología.")
        return diccionario
    except Exception as e:
        print(f"Error al cargar el diccionario: {e}")
        return pd.DataFrame(columns=['Palabra', 'Frecuencia', 'Frecuencia_Limpia'])

def cargar_listas_sentimientos():
    """
    Carga las listas de palabras positivas, negativas y neutrales para análisis de sentimiento,
    filtrando solamente las entradas relacionadas con astrología.

    Esta función lee tres archivos CSV con palabras clasificadas según su sentimiento,
    filtra solo las filas donde Pseudociencia es 'astrologia' y convierte las palabras
    en conjuntos para facilitar la búsqueda.

    Returns:
        tuple: Tupla con tres conjuntos (set) de palabras (positivas, negativas, neutrales)
    """
    print("Cargando listas de sentimientos...")

    try:
        # Definir rutas a los archivos de sentimientos
        positivas_path = os.path.join(BASE_DIR, 'AnalisisPositivo.csv')
        negativas_path = os.path.join(BASE_DIR, 'AnalisisNegativo.csv')
        neutrales_path = os.path.join(BASE_DIR, 'AnalisisNeutral.csv')

        print(f"Cargando archivo positivo: {positivas_path}")
        print(f"Cargando archivo negativo: {negativas_path}")
        print(f"Cargando archivo neutral: {neutrales_path}")

        # Cargar los archivos CSV
        positivas = pd.read_csv(positivas_path, encoding='latin1')
        negativas = pd.read_csv(negativas_path, encoding='latin1')
        neutrales = pd.read_csv(neutrales_path, encoding='latin1')

        # Filtrar solo las filas relacionadas con astrología
        positivas = positivas[positivas['Pseudociencia'].str.lower() == 'astrologia']
        negativas = negativas[negativas['Pseudociencia'].str.lower() == 'astrologia']
        neutrales = neutrales[neutrales['Pseudociencia'].str.lower() == 'astrologia']

        # Limpiar y procesar frecuencias
        def limpiar_frecuencia(valor):
            if pd.isna(valor) or valor == '':
                return 0
            # Si ya es un número, devolverlo como está
            if isinstance(valor, (int, float)):
                return float(valor)
            # Convertir a string y limpiar comillas y comas
            valor_str = str(valor).replace('"', '').replace(',', '')
            # Intentar convertir a float
            try:
                return float(valor_str)
            except ValueError:
                return 0

        # Añadir columnas de frecuencia limpia
        if 'Frecuencia' in positivas.columns:
            positivas['Frecuencia_Limpia'] = positivas['Frecuencia'].apply(limpiar_frecuencia)
            positivas = positivas.sort_values('Frecuencia_Limpia', ascending=False)

        if 'Frecuencia' in negativas.columns:
            negativas['Frecuencia_Limpia'] = negativas['Frecuencia'].apply(limpiar_frecuencia)
            negativas = negativas.sort_values('Frecuencia_Limpia', ascending=False)

        if 'Frecuencia' in neutrales.columns:
            neutrales['Frecuencia_Limpia'] = neutrales['Frecuencia'].apply(limpiar_frecuencia)
            neutrales = neutrales.sort_values('Frecuencia_Limpia', ascending=False)

        # Extraer solo las palabras y convertirlas a conjuntos para búsqueda eficiente
        palabras_positivas = set(positivas['Palabra'].dropna().astype(str).str.lower().values)
        palabras_negativas = set(negativas['Palabra'].dropna().astype(str).str.lower().values)
        palabras_neutrales = set(neutrales['Palabra'].dropna().astype(str).str.lower().values)

        # Mostrar estadísticas de palabras cargadas
        print(f"Palabras positivas para astrología: {len(palabras_positivas)}")
        print(f"Palabras negativas para astrología: {len(palabras_negativas)}")
        print(f"Palabras neutrales para astrología: {len(palabras_neutrales)}")

        # Guardar las listas procesadas para referencia
        positivas.to_csv(os.path.join(BASE_DIR, 'astrologia_palabras_positivas.csv'), index=False)
        negativas.to_csv(os.path.join(BASE_DIR, 'astrologia_palabras_negativas.csv'), index=False)
        neutrales.to_csv(os.path.join(BASE_DIR, 'astrologia_palabras_neutrales.csv'), index=False)
        print("Listas de palabras guardadas con frecuencias procesadas.")

        return palabras_positivas, palabras_negativas, palabras_neutrales

    except Exception as e:
        print(f"Error al cargar listas de sentimientos: {e}")
        return set(), set(), set()

def limpiar_texto(texto):
    """
    Limpia y normaliza el texto para el análisis.

    Proceso de limpieza:
    1. Convertir a minúsculas
    2. Eliminar URLs
    3. Eliminar menciones (@usuario) y hashtags (#tema)
    4. Eliminar caracteres especiales
    5. Eliminar números
    6. Eliminar espacios múltiples

    Args:
        texto (str): Texto a limpiar

    Returns:
        str: Texto limpio y normalizado
    """
    if pd.isna(texto):
        return ""

    # Convertir a minúsculas
    texto = texto.lower()

    # Eliminar URLs
    texto = re.sub(r'https?://\S+', '', texto)

    # Eliminar menciones y hashtags
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'#\w+', '', texto)

    # Eliminar caracteres especiales y mantener letras, números y espacios
    texto = re.sub(r'[^\w\s]', '', texto)

    # Eliminar números
    texto = re.sub(r'\d+', '', texto)

    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto

def clasificar_tweets(tweets_df, palabras_positivas, palabras_negativas, palabras_neutrales):
    """
    Clasifica tweets como positivos, negativos o neutrales basado en el diccionario.

    Para cada tweet:
    1. Limpia el texto
    2. Cuenta ocurrencias de palabras positivas, negativas y neutrales
    3. Asigna etiqueta de sentimiento según el recuento mayor

    Args:
        tweets_df (pandas.DataFrame): DataFrame con tweets a clasificar
        palabras_positivas (set): Conjunto de palabras positivas
        palabras_negativas (set): Conjunto de palabras negativas
        palabras_neutrales (set): Conjunto de palabras neutrales

    Returns:
        pandas.DataFrame: DataFrame con tweets clasificados y métricas adicionales
    """
    print("Clasificando tweets según sentimiento...")

    resultados = []

    # Procesar cada tweet
    for _, row in tweets_df.iterrows():
        tweet_id = row['Tweet Id']
        texto = limpiar_texto(row['Text'])

        # Tokenizar el tweet en palabras
        palabras = texto.split()

        # Contar ocurrencias de palabras positivas, negativas y neutrales
        count_pos = sum(1 for palabra in palabras if palabra in palabras_positivas)
        count_neg = sum(1 for palabra in palabras if palabra in palabras_negativas)
        count_neu = sum(1 for palabra in palabras if palabra in palabras_neutrales)

        # Asignar etiqueta según conteo
        if count_pos > count_neg:
            sentiment = 'Positivo'
        elif count_neg > count_pos:
            sentiment = 'Negativo'
        else:
            sentiment = 'Neutral'

        # Almacenar resultados para este tweet
        resultados.append({
            'Tweet_Id': tweet_id,
            'Texto': row['Text'],
            'Texto_Limpio': texto,
            'Palabras_Positivas': count_pos,
            'Palabras_Negativas': count_neg,
            'Palabras_Neutrales': count_neu,
            'Sentimiento': sentiment
        })

    # Crear DataFrame con resultados
    resultados_df = pd.DataFrame(resultados)

    # Mostrar estadísticas de clasificación
    counts = resultados_df['Sentimiento'].value_counts()
    print(f"Tweets positivos: {counts.get('Positivo', 0)}")
    print(f"Tweets negativos: {counts.get('Negativo', 0)}")
    print(f"Tweets neutrales: {counts.get('Neutral', 0)}")

    return resultados_df

def aplicar_kmeans(resultados_df, n_clusters=3):
    """
    Aplica K-means clustering para agrupar tweets similares.

    Proceso:
    1. Vectoriza los textos usando TF-IDF
    2. Aplica algoritmo K-means para agrupar en clusters
    3. Identifica palabras relevantes para cada cluster

    Args:
        resultados_df (pandas.DataFrame): DataFrame con tweets clasificados
        n_clusters (int): Número de clusters a crear

    Returns:
        tuple: (DataFrame actualizado con etiquetas de cluster, diccionario con información de clusters)
    """
    print(f"Aplicando K-means con {n_clusters} clusters...")

    # Usar TF-IDF para vectorizar los textos
    # TF-IDF pondera más las palabras importantes y menos las comunes
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=['de', 'la', 'el', 'en', 'y', 'a', 'que', 'un', 'es', 'no', 'por', 'los', 'las'])
    X = vectorizer.fit_transform(resultados_df['Texto_Limpio'])

    # Aplicar K-means para agrupar textos similares
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    resultados_df['Cluster'] = kmeans.fit_predict(X)

    # Analizar palabras relevantes para cada cluster
    feature_names = vectorizer.get_feature_names_out()
    centroids = kmeans.cluster_centers_

    # Recopilar información sobre cada cluster
    clusters_info = {}
    for i in range(n_clusters):
        # Obtener índices de las palabras más relevantes para este cluster
        indices = centroids[i].argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in indices]
        clusters_info[i] = top_words

        # Mostrar estadísticas del cluster
        cluster_size = (resultados_df['Cluster'] == i).sum()
        print(f"Cluster {i} - Tamaño: {cluster_size} tweets")
        print(f"Palabras más relevantes: {', '.join(top_words)}")

    return resultados_df, clusters_info

def aplicar_naive_bayes(resultados_df):
    """
    Aplica clasificación Naive Bayes para predecir sentimiento en tweets.

    Proceso:
    1. Prepara datos de entrenamiento y prueba
    2. Vectoriza textos con CountVectorizer
    3. Entrena clasificador Naive Bayes
    4. Evalúa precisión del modelo

    Args:
        resultados_df (pandas.DataFrame): DataFrame con tweets clasificados

    Returns:
        tuple: (modelo entrenado, vectorizador utilizado)
    """
    print("Aplicando clasificación Naive Bayes...")

    # Verificar que haya suficientes datos para cada clase
    class_counts = resultados_df['Sentimiento'].value_counts()
    if len(class_counts) < 2 or min(class_counts) < 5:
        print("No hay suficientes datos para entrenar un clasificador Naive Bayes.")
        return None, None

    # Preparar datos para entrenamiento
    X = resultados_df['Texto_Limpio']
    y = resultados_df['Sentimiento']

    # Vectorizar los textos (convertir palabras en frecuencias)
    vectorizer = CountVectorizer(max_features=1000)
    X_vectorized = vectorizer.fit_transform(X)

    # Dividir en conjuntos de entrenamiento (70%) y prueba (30%)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

    # Entrenar clasificador Naive Bayes
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Evaluar rendimiento del modelo
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del clasificador Naive Bayes: {accuracy:.2f}")
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))

    return classifier, vectorizer

def generar_visualizaciones(resultados_df, clusters_info):
    """
    Genera visualizaciones para el análisis de tweets.

    Crea:
    1. Gráfico de barras de distribución de sentimientos
    2. Gráfico circular de proporción de sentimientos
    3. Nubes de palabras por sentimiento
    4. Gráfico de distribución de clusters
    5. Nubes de palabras por cluster

    Args:
        resultados_df (pandas.DataFrame): DataFrame con tweets clasificados
        clusters_info (dict): Diccionario con información de clusters
    """
    print("Generando visualizaciones...")

    # Crear directorio para gráficas si no existe
    grafica_dir = os.path.join(BASE_DIR, 'graficas')
    if not os.path.exists(grafica_dir):
        os.makedirs(grafica_dir)

    # 1. Gráfico de barras: Distribución de sentimientos
    plt.figure(figsize=(10, 6))
    counts = resultados_df['Sentimiento'].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values, palette=['green', 'gray', 'red'])
    plt.title('Distribución de Tweets por Sentimiento')
    plt.xlabel('Sentimiento')
    plt.ylabel('Número de Tweets')

    # Añadir valores encima de las barras
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(grafica_dir, 'distribucion_sentimientos.png'))
    plt.close()

    # 2. Pie chart: Proporción de sentimientos
    plt.figure(figsize=(8, 8))
    colors = ['green', 'gray', 'red']
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Proporción de Tweets por Sentimiento')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(grafica_dir, 'proporcion_sentimientos.png'))
    plt.close()

    # 3. Nubes de palabras por sentimiento
    sentimientos = ['Positivo', 'Negativo', 'Neutral']
    for sentimiento in sentimientos:
        subset = resultados_df[resultados_df['Sentimiento'] == sentimiento]
        if len(subset) > 0:
            text = ' '.join(subset['Texto_Limpio'])
            if text.strip():
                # Generar nube de palabras
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                     max_words=100, contour_width=3, contour_color='steelblue')
                wordcloud.generate(text)

                # Guardar imagen
                plt.figure(figsize=(10, 7))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Nube de Palabras - Tweets {sentimiento}s')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(grafica_dir, f'nube_palabras_{sentimiento.lower()}.png'))
                plt.close()

    # 4. Visualización de clusters
    if clusters_info:
        plt.figure(figsize=(12, 8))
        cluster_sizes = resultados_df['Cluster'].value_counts().sort_index()
        ax = sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
        plt.title('Distribución de Tweets por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Número de Tweets')

        # Añadir valores encima de las barras
        for i, v in enumerate(cluster_sizes.values):
            ax.text(i, v + 5, str(v), ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(grafica_dir, 'distribucion_clusters.png'))
        plt.close()

        # 5. Nubes de palabras por cluster
        for cluster_id, top_words in clusters_info.items():
            subset = resultados_df[resultados_df['Cluster'] == cluster_id]
            if len(subset) > 0:
                text = ' '.join(subset['Texto_Limpio'])
                if text.strip():
                    # Generar nube de palabras
                    wordcloud = WordCloud(width=800, height=400, background_color='white',
                                         max_words=100, contour_width=3, contour_color='steelblue')
                    wordcloud.generate(text)

                    # Guardar imagen
                    plt.figure(figsize=(10, 7))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.title(f'Nube de Palabras - Cluster {cluster_id}')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(grafica_dir, f'nube_palabras_cluster_{cluster_id}.png'))
                    plt.close()

def main():
    """
    Función principal que ejecuta todo el proceso de análisis.

    Secuencia de ejecución:
    1. Cargar datos (tweets y diccionarios)
    2. Clasificar tweets por sentimiento
    3. Aplicar K-means para clustering
    4. Entrenar clasificador Naive Bayes
    5. Generar visualizaciones
    6. Guardar resultados
    """
    print("=== ANÁLISIS DE TWEETS DE ASTROLOGÍA ===")

    # 1. Cargar datos
    tweets_df = cargar_tweets()
    palabras_positivas, palabras_negativas, palabras_neutrales = cargar_listas_sentimientos()

    # Cargar diccionario general de astrología para enriquecer el análisis
    diccionario_astrologia = cargar_diccionario()

    # Guardar las primeras 100 palabras más frecuentes del diccionario
    top_palabras = diccionario_astrologia.head(100)
    top_palabras.to_csv(os.path.join(BASE_DIR, 'top_palabras_astrologia.csv'), index=False)
    print(f"Top 100 palabras de astrología guardadas en 'top_palabras_astrologia.csv'")

    if len(tweets_df) == 0:
        print("No hay tweets para analizar.")
        return

    # 2. Clasificar tweets por sentimiento
    resultados_df = clasificar_tweets(tweets_df, palabras_positivas, palabras_negativas, palabras_neutrales)

    # 3. Aplicar K-means para agrupar tweets
    resultados_df, clusters_info = aplicar_kmeans(resultados_df, n_clusters=3)

    # 4. Entrenar un clasificador Naive Bayes
    classifier, vectorizer = aplicar_naive_bayes(resultados_df)

    # 5. Generar visualizaciones
    generar_visualizaciones(resultados_df, clusters_info)

    # 6. Guardar resultados en CSV
    resultados_path = os.path.join(BASE_DIR, 'resultados_analisis.csv')
    resultados_df.to_csv(resultados_path, index=False)
    print(f"\n¡Análisis completado! Los resultados se han guardado en '{resultados_path}'")
    print(f"Las visualizaciones se han guardado en el directorio '{os.path.join(BASE_DIR, 'graficas')}'")

# Punto de entrada del script
if __name__ == "__main__":
    main()