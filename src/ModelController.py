"""
ModelController.py

Carga los 3 artefactos entrenados (vectorizador TF-IDF, SVD/LSA y el modelo
Random Forest) y expone una función para predecir el ODS de textos nuevos.
"""

import os
import joblib
import pandas as pd

from src.DataPrePreprocessing import pipeline_preparacion, text_preprocess, descargar_recursos_nltk

# Ruta base de los artefactos entrenados (relativa a la raíz del proyecto)
RUTA_MODELOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "models")


class ModelController:
    def __init__(self):
        # Asegura que los recursos de NLTK estén disponibles antes de usar el vectorizador
        descargar_recursos_nltk()

        self.vectorizer = joblib.load(os.path.join(RUTA_MODELOS, "tfidf_vectorizer.joblib"))
        self.svd = joblib.load(os.path.join(RUTA_MODELOS, "svd_lsa.joblib"))
        self.model = joblib.load(os.path.join(RUTA_MODELOS, "model.joblib"))

    def predecir(self, textos: pd.Series) -> pd.DataFrame:
        """
        Recibe una Serie/lista de textos crudos y retorna un DataFrame con
        el texto original, el ODS predicho y (si el modelo lo soporta) la
        probabilidad asociada a la clase predicha.
        """
        # 1. Empaquetar en un DataFrame para reutilizar el pipeline de limpieza
        df = pd.DataFrame({"textos": list(textos)})

        # 2. Limpieza básica (minúsculas, tildes, caracteres especiales, números)
        df_procesado = pipeline_preparacion.transform(df)

        # 3. Preprocesamiento de texto (tokenización, stopwords, lematización) —
        #    el vectorizador se entrenó sobre texto YA pasado por text_preprocess,
        #    así que hay que aplicarlo aquí a mano antes de vectorizar.
        textos_preprocesados = df_procesado["textos"].apply(text_preprocess)

        # 4. Vectorización TF-IDF
        tfidf_matrix = self.vectorizer.transform(textos_preprocesados)

        # 5. Reducción de dimensionalidad (LSA / TruncatedSVD)
        x_lsa = self.svd.transform(tfidf_matrix)

        # 6. Predicción
        predicciones = self.model.predict(x_lsa)

        resultado = pd.DataFrame({
            "texto": textos.values if hasattr(textos, "values") else list(textos),
            "ODS_predicho": predicciones,
        })

        if hasattr(self.model, "predict_proba"):
            probabilidades = self.model.predict_proba(x_lsa)
            resultado["confianza"] = probabilidades.max(axis=1)

        return resultado
