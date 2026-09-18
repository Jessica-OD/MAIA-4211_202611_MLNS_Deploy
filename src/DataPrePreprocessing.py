"""
DataPrePreprocessing.py
Replica EXACTAMENTE las funciones de limpieza de texto usadas durante el entrenamiento (notebook Solución_Microproyecto2).
"""

import pandas.api.types as ptypes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Funciones de limpieza a nivel de DataFrame (usadas en el pipeline inicial)

def a_minusculas(texto):
    """Convierte a minúsculas todas las columnas de texto de un DataFrame"""
    for col in texto.columns:
        if ptypes.is_string_dtype(texto[col]) or ptypes.is_object_dtype(texto[col]):
            texto[col] = texto[col].str.lower()
    return texto


def remover_tildes(texto):
    """Elimina tildes. Funciona tanto sobre un string como sobre un DataFrame"""
    # Crea un mapa de traducción que se utiliza para eliminar acentos/tíldes
    tabla_acentos = str.maketrans('áéíóúüÁÉÍÓÚÜ', 'aeiouuAEIOUU')
    if isinstance(texto, str):
        return texto.translate(tabla_acentos)
    else:
        for col in texto.columns:
            if ptypes.is_string_dtype(texto[col]) or ptypes.is_object_dtype(texto[col]):
                texto[col] = texto[col].astype(str).str.translate(tabla_acentos)
        return texto


def eliminar_caracteres_especiales(texto):
    """Elimina signos de puntuación y caracteres especiales"""
    for col in texto.columns:
        if ptypes.is_string_dtype(texto[col]) or ptypes.is_object_dtype(texto[col]):
            # Busca cualquier carácter que NO este en el listado (negación) y lo reemplaza por espacio en blanco
            texto[col] = texto[col].str.replace(r'[^a-z0-9ñ\s]', ' ', regex=True)
            # Busca uno o más espacios en blanco seguidos y los reemplaza por un espacio en blanco
            texto[col] = texto[col].str.strip().str.replace(r'\s+', ' ', regex=True)
    return texto


def eliminar_numeros(texto):
    """Elimina dígitos numéricos"""
    for col in texto.columns:
        if ptypes.is_string_dtype(texto[col]) or ptypes.is_object_dtype(texto[col]):
            texto[col] = texto[col].str.replace(r'\d', ' ', regex=True)
            texto[col] = texto[col].str.strip().str.replace(r'\s+', ' ', regex=True)
    return texto


# Pipeline de preparación a nivel de DataFrame (limpieza básica)
pipeline_preparacion = Pipeline([
    ("a_minusculas", FunctionTransformer(a_minusculas)),
    ("remover_tildes", FunctionTransformer(remover_tildes)),
    ("eliminar_caracteres_especiales", FunctionTransformer(eliminar_caracteres_especiales)),
    ("eliminar_numeros", FunctionTransformer(eliminar_numeros)),
])

# Preprocesamiento a nivel de texto individual (usado dentro del TfidfVectorizer)

def text_preprocess(text):
    """Tokeniza, elimina stopwords en español y lematiza"""
    # Inicializa extractor que busca secuencia de caracteres alfanuméricos, ignorando automáticamente cualquier signo de puntuación
    tokenizer = RegexpTokenizer(r'\w+')
    # Inicializa reductor de palabras (reducción a su forma base)
    lemmatizer = WordNetLemmatizer()

    # Convierte el string en una lista de palabras individuales
    tokens = tokenizer.tokenize(text)
    # Remueve tíldes a las palabras que sobreviven el filtro de stopwords
    tokens = [remover_tildes(word) for word in tokens if word not in stopwords.words('spanish')]
    # Pasa por el lematizador para convertir las palabras a su forma base
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Elimina todas las palabras que tienen menos de 3 caracteres
    tokens = [word for word in tokens if len(word) >= 3]

    # Junta de nuevo todas las palabras limpias en una sola cadena de texto (string)
    return ' '.join(tokens)


def descargar_recursos_nltk():
    """Descarga (una sola vez) los recursos de NLTK necesarios en el entorno de despliegue"""
    import nltk
    for recurso in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        try:
            nltk.download(recurso, quiet=True)
        except Exception:
            pass