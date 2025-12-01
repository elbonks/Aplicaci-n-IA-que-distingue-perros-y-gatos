import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Reconocimiento Perros vs Gatos", page_icon="🐾")

st.title("🐶 Detector de Mascotas 🐱")
st.write("Usa la cámara para saber si es un perro o un gato.")

@st.cache_resource
def carga_modelo():
    modelo = keras.models.load_model("app/keras_model.h5", compile=False)
    clases = open("app/labels.txt", "r").readlines()
    return modelo, clases

try:
    mi_modelo, nombre_clases = carga_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

imagen_camara = st.camera_input("Haz una foto")

if imagen_camara is not None:
    # Preparar imagen
    imagen = Image.open(imagen_camara).convert("RGB")
    imagen = ImageOps.fit(imagen, (224, 224), Image.Resampling.LANCZOS)
    imagen_array = np.asarray(imagen)
    normalizada = (imagen_array.astype(np.float32) / 127.5) - 1

    lote_imagenes = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    lote_imagenes[0] = normalizada

    # Predicción
    resultados = mi_modelo.predict(lote_imagenes)
    indice = np.argmax(resultados[0])
    etiqueta = nombre_clases[indice].strip()
    probabilidad = resultados[0][indice]

    st.divider()

    if "Perro" in etiqueta:
        st.success("¡Es un **PERRO**! 🐶")
        st.balloons()
    else:
        st.success("¡Es un **GATO**! 🐱")
        st.snow()

    st.write(f"Estoy un {probabilidad:.2%} seguro.")
