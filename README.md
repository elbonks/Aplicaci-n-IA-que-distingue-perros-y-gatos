# Aplicaci-n-IA-que-distingue-perros-y-gatos
esta es una IA entrenada mediante teachable machine para que distinga entre gatos y perros. Esta ha sido más tarde programada mediante google colab para hacer que sea una aplicación funcional para cualquier imagen de perros y gatos.

Descripción Detallada del Modelo:

Este cuaderno implementa una aplicación de Machine Learning de bajo código para la clasificación de imágenes de "Perro/Gato". El modelo utilizado fue entrenado previamente en Teachable Machine y exportado en formato Keras.

Componentes Clave:
Carga del Modelo: Se carga un modelo Keras (keras_model.h5) previamente entrenado. Es importante destacar que se activa el modo Keras 2 legacy (os.environ['TF_USE_LEGACY_KERAS'] = '1') para asegurar la compatibilidad con el formato de exportación de Teachable Machine. El modelo se carga sin compilación (compile=False) ya que solo se utilizará para inferencia. El modelo tiene una arquitectura que procesa imágenes y emite una predicción de dos clases.

Preprocesamiento de Imágenes: Antes de que una imagen pueda ser clasificada, debe ser transformada para que coincida con el formato de las imágenes de entrenamiento. Esto incluye:

Redimensionamiento: La imagen se ajusta a un tamaño de 224x224 píxeles.
Normalización: Los valores de los píxeles (originalmente en un rango de [0, 255]) se normalizan a un rango de [-1, 1]. Esto se logra dividiendo por 127.5 y restando 1.
Preparación del Lote: Aunque se procese una sola imagen, el modelo espera entradas en formato de lote (batch). Por lo tanto, la imagen preprocesada se coloca dentro de un array con una dimensión de lote.
Inferencia (Predicción): El modelo cargado (mi_modelo) utiliza el método predict() para clasificar la imagen preprocesada. El resultado es un array de probabilidades, donde cada elemento corresponde a la probabilidad de que la imagen pertenezca a una de las clases definidas.

Interpretación de Resultados: Se identifica el índice con la mayor probabilidad (np.argmax) para determinar la clase predicha (por ejemplo, 'gato' o 'perro'). La probabilidad asociada a esa clase es la confianza del modelo en su predicción.

Evaluación del Modelo: Se realiza una fase de prueba donde el modelo clasifica un conjunto de imágenes de prueba (contenidas en la ruta /content/drive/MyDrive/cat vs dog/test). Para cada imagen, se compara la etiqueta predicha con la etiqueta esperada. Se calcula la precisión (número de aciertos / total de predicciones) y la probabilidad media de los aciertos, además de registrar las predicciones incorrectas para un análisis posterior.
