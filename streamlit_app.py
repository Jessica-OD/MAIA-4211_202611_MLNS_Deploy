import streamlit as st
import pandas as pd

from src.ModelController import ModelController

# Nombres oficiales de los 17 ODS, solo para que la interfaz sea más legible
NOMBRES_ODS = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de las desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos",
}

st.set_page_config(page_title="Clasificador de textos por ODS", layout="wide")

st.title("Clasificador de textos por Objetivo de Desarrollo Sostenible (ODS)")
st.write(
    "Sube un archivo con textos (CSV o Excel, con una columna llamada "
    "**textos**) o escribe un texto directamente para identificar a qué "
    "ODS está relacionado, según el modelo Random Forest entrenado."
)


@st.cache_resource
def cargar_controlador():
    return ModelController()


controller = cargar_controlador()

# --------------------------------------------------------------
# Sección 1: entrada de datos
# --------------------------------------------------------------
st.header("1. Datos de entrada")

modo = st.radio("¿Cómo quieres ingresar el texto?", ["Escribir un texto", "Subir archivo (CSV/Excel)"])

df_entrada = None

if modo == "Escribir un texto":
    texto_manual = st.text_area("Escribe el texto a clasificar", height=150)
    if texto_manual.strip():
        df_entrada = pd.DataFrame({"textos": [texto_manual]})
else:
    archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])
    if archivo is not None:
        if archivo.name.endswith(".csv"):
            df_entrada = pd.read_csv(archivo)
        else:
            df_entrada = pd.read_excel(archivo)

        if "textos" not in df_entrada.columns:
            st.error("El archivo debe contener una columna llamada 'textos'.")
            df_entrada = None
        else:
            st.dataframe(df_entrada.head())

# --------------------------------------------------------------
# Sección 2: resultados
# --------------------------------------------------------------
st.header("2. Resultado de la predicción")

if df_entrada is not None and st.button("Clasificar"):
    with st.spinner("Procesando texto(s) y prediciendo..."):
        resultado = controller.predecir(df_entrada["textos"])
        resultado["ODS_nombre"] = resultado["ODS_predicho"].map(NOMBRES_ODS)

    st.success("Clasificación completada")
    st.dataframe(resultado)

    if len(resultado) == 1:
        fila = resultado.iloc[0]
        st.metric(
            label="ODS predicho",
            value=f"ODS {fila['ODS_predicho']} – {fila['ODS_nombre']}",
        )
        if "confianza" in resultado.columns:
            st.caption(f"Confianza del modelo: {fila['confianza']:.1%}")
else:
    st.info("Ingresa un texto o sube un archivo y presiona 'Clasificar'.")
