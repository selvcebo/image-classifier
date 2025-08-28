# 🧠 Clasificador de Dígitos con CNN
Proyecto que implementa una red neuronal convolucional (CNN) en TensorFlow/Keras para clasificar imágenes de dígitos (dataset load_digits de scikit-learn) y reconocer dígitos dibujados a mano a partir de imágenes externas.


## 📌 Características
- Entrenamiento sobre el dataset load_digits (8×8 px, escala de grises).
- Preprocesamiento de imágenes externas:
- Lectura y normalización a escala 0–16.
- Cuadrado y recorte centrado.
- Reescalado a 8×8 px.
- Evaluación con métricas: Accuracy, Precision, Recall, F1-score y Matriz de confusión.
- Visualización de cada etapa para interpretabilidad.
-Clasificación de varios archivos en lote, mostrando predicciones en consola y gráficamente.

## 📂 Estructura de Archivos
```
│-- main.py              # Script principal con entrenamiento, evaluación y funciones
│-- digito*.jpg          # Imágenes de prueba
│-- requirements.txt     # Dependencias
│-- README.md            # Este documento
```
## 🛠 Requisitos
- Python 3.8+
- TensorFlow
- scikit-learn
- numpy
- matplotlib
- seaborn
- Pillow
- scikit-image

Instalar dependencias:

bash
```
pip install -r requirements.txt
```

## 🚀 Ejecución
Clonar el repositorio y entrar a la carpeta del proyecto:

bash
```
git clone git@github.com:selvcebo/image-classifier.git

```
Colocar tus imágenes .jpg en la carpeta del proyecto.

Ejecutar:

bash
```
python main.py
```
## El script:
- Entrena el modelo.
- Muestra métricas y matriz de confusión.
- Procesa tus imágenes externas y predice sus dígitos.

## 🧩 Pipeline de Clasificación Externa
- leer_digito() Convierte imagen a escala de grises (0–16) y binariza (<=7 → 0, >7 → 16).
- recortar_cuadrado() Recorta el dígito centrado según píxeles > 0.
- reescalar_8x8() Reduce a 8×8 px manteniendo la escala 0–16.
- predecir_digito() Ajusta la imagen al shape (1, 8, 8, 1) y devuelve el dígito más probable.

## 📊 Ejemplo de Salida
```
Accuracy: 0.987
Precision: 0.987
Recall: 0.987
F1: 0.987

Predicciones para las imágenes cargadas:
digito_felipe.jpg: 3
digito_laura.jpg: 8
...
```
(Incluye también visualizaciones paso a paso y matriz de confusión.)

## 💡 Notas
Puedes guardar y cargar el modelo con model.save() y load_model() para no reentrenar cada vez.

Evita usar imágenes demasiado pequeñas o con mucho ruido: el recorte y la binarización funcionan mejor con contraste claro.

Ajusta epochs y batch_size según el rendimiento que busques.

## 📜 Licencia
Distribuido bajo licencia MIT. Si reutilizas este código, dale una estrella ⭐ en GitHub.

---

## ✨ Autor
Sergio Esteban León Valencia | Desarrollador Fullstack | Entusiasta de IA y Machine Learning 
