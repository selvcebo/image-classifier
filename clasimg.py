# 1. Importar dependencias.
# 1. Improt dependencies.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from PIL import Image
from skimage.transform import resize
warnings.filterwarnings("ignore")

# 2. Cargar la dataset de imágenes de dígitos identificando la información de características X y la objetivo y.
# 2. Load the digit image dataset identifying feature information X and target y.
mis_digitos = load_digits()

# Tu array
# Your array
indice = 1113
mis_digitos.images[indice]
img = mis_digitos.images[indice]

# Normalizar los valores al rango 0–255 para escala de grises
# Normalize values to the 0-255 range for grayscale
normalized = (1 - (img / 16)) * 255

plt.imshow(normalized, cmap='gray', vmin=0, vmax=255)
plt.axis('off')  # Quitar ejes
plt.show()

X = mis_digitos.images
y = mis_digitos.target

# 3. Definimos las datas de entrenamiento  Xtrain  y  ytrain, de validación  Xval  y  yval  y de prueba  Xtest  y  ytest.
# 3. We define the training data Xtrain and ytrain, validation Xval and yval, and test Xtest and ytest.
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=53)
X.shape, X_train_val.shape, X_test.shape,y.shape, y_train_val.shape, y_test.shape
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=53)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# 4. Hacemos ajuste a las imágenes  X.
# 4. We adjust the images X.
X_train_ajustado = X_train.reshape(X_train.shape[0], 8, 8, 1)
X_val_ajustado = X_val.reshape(X_val.shape[0], 8, 8, 1)
X_test_ajustado = X_test.reshape(X_test.shape[0], 8, 8, 1)

# 5. Hacemos one hot de clases  y.
# 5. We do one hot of classes y.
np.unique(y)
y_train_one_hot = to_categorical(y_train)
y_val_one_hot = to_categorical(y_val)
y_test_one_hot = to_categorical(y_test)

# 6. Creamos un modelo de red neuronal convolucional para clasificar los dígitos.
# 6. We create a convolutional neural network model to classify the digits.
mi_modelo = tf.keras.Sequential()
mi_modelo.add(Conv2D(24, kernel_size=3, padding='same', activation='relu', input_shape=(8, 8, 1)))
mi_modelo.add(MaxPooling2D())
mi_modelo.add(Conv2D(48, kernel_size=3, padding='same', activation='relu'))
mi_modelo.add(MaxPooling2D())
mi_modelo.add(Flatten())
mi_modelo.add(Dense(10, activation='softmax'))
mi_modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mi_modelo.summary()

# 7. Hacemos entrenamiento y validación de  Xtrain,  ytrain  y  Xval  y  yval.
# 7. We do training and validation of Xtrain, ytrain and Xval and yval.
mi_historia = mi_modelo.fit(X_train_ajustado, y_train_one_hot, batch_size=512, epochs=100, validation_data=(X_val_ajustado, y_val_one_hot))
print("Entrenamiento completado")

# 8. Hacemos pruebas de  Xtest ,  ytest , con scores y matriz de confusión.
# 8. We do tests of Xtest, ytest, with scores and confusion matrix.
y_pred_one_hot = mi_modelo.predict(X_test_ajustado)
y_pred_one_hot[:1]

""" ---------- """

plt.imshow(X_test[0], cmap=plt.cm.gray_r)
plt.show()


y_test[0]

y_pred = np.argmax(y_pred_one_hot, axis=1)
y_pred[:2]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy: {accuracy} \n Precision: {precision} \n Recall: {recall} \n F1: {f1}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 9. Clasificar con el modelo de IA dígitos que hagamos nosotros.


# Fases

# 1. Función leer_digito(archivo.jpg) en una imagen en escala de grises con niveles, 0 para blanco hasta 16 para negro.
# 2. Función cuadrar_imagen(img) haciendo un recorte cuadrado desde el centro de la imagen.
# 3. Función reescalar_imagen(img) haciendo que img pase de NxN píxeles a 8x8 píxeles.
# 4. Función predecir(img) que hace el reshape y el proceso de predicción con nuestra mi_modelo.

# 9. Classify digits we make ourselves with the AI model.


# Phases

# 1. Function leer_digito(archivo.jpg) to read an image in grayscale with levels, 0 for white up to 16 for black.
# 2. Function cuadrar_imagen(img) that crops a square from the center of the image.
# 3. Function reescalar_imagen(img) that resizes img from NxN pixels to 8x8 pixels.
# 4. Function predecir(img) that reshapes and predicts with our mi_modelo.





def leer_digito(archivo_jpg):
    # 1. Función leer_digito(archivo.jpg) en una imagen en escala de grises con niveles, 0 para blanco hasta 16 para negro.
    # 1. Function leer_digito(archivo.jpg) to read an image in grayscale with levels, 0 for white up to 16 for black.
    img = Image.open(archivo_jpg).convert("L")
    img_array = np.array(img)

    # Normalizar la imagen al rango 0-16
    # Normalize the image to the range 0-16
    normalized_img = (1 - (img_array/ 255.0)) * 16

    # Binarizar la imagen: valores <=7 a 0, >7 a 16
    # Binarize the image: values <=7 to 0, >7 to 16
    normalized_img[normalized_img <= 7] = 0
    normalized_img[normalized_img > 7] = 16
    return normalized_img

image_files = [
        "digito felipe.jpg", "digito laura.jpg", "digito miguel.jpg", "digito oscar.jpg", "digito quiros.jpg", "digito sergio.jpg"
]

plt.figure(figsize=(15, 5))

for i, img_file in enumerate(image_files):
    img = leer_digito(img_file)
    if img is not None:
        # Mostrar la imagen normalizada
        # Show the normalized image
        plt.subplot(1, len(image_files), i + 1)
        plt.imshow(img, cmap=plt.cm.gray_r)
        plt.axis("off")
        plt.title(f'{img_file}')

plt.tight_layout()
plt.show()

def cuadrar_imagen(img):
    # 2. Función cuadrar_imagen(img) haciendo un recorte cuadrado desde el centro de la imagen.
    # 2. Function cuadrar_imagen(img) that crops a square from the center of the image.
    height, width = img.shape
    max_dim = max(height, width)

    pad_h = max_dim - height
    pad_w = max_dim - width

    # Rellenar la imagen para hacerla cuadrada
    # Pad the image to make it square
    squared_img = np.pad(img,((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    return squared_img

plt.figure(figsize=(15, 5))

for i, img_file in enumerate(image_files):
    img = leer_digito(img_file)
    if img is not None:
        squared_img = cuadrar_imagen(img)
        # Mostrar la imagen cuadrada
        # Show the squared image
        plt.subplot(1, len(image_files), i + 1)
        plt.imshow(squared_img, cmap=plt.cm.gray_r)
        plt.axis("off")
        plt.title(f'{img_file}')

plt.tight_layout

def recortar_cuadrado(img):
    # 3. Función recortar_cuadrado(img) haciendo un recorte cuadrado centrado en el dígito.
    # 3. Function recortar_cuadrado(img) cropping a square centered on the digit.
    coords = np.argwhere(img > 0)

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    height = x_max - x_min + 1
    width = y_max - y_min + 1

    crop_size = max(height, width)

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    crop_x_min = max(0, center_x - crop_size // 2)
    crop_x_max = min(img.shape[0], center_x + crop_size // 2 + (crop_size % 2))
    crop_y_min = max(0, center_y - crop_size // 2)
    crop_y_max = min(img.shape[1], center_y + crop_size // 2 + (crop_size % 2))

    actual_crop_size = min(crop_x_max - crop_x_min, crop_y_max - crop_y_min)
    crop_x_max = crop_x_min + actual_crop_size
    crop_y_max = crop_y_min + actual_crop_size

    # Recortar la imagen centrada en el dígito
    # Crop the image centered on the digit
    cropped_img = img[crop_x_min:crop_x_max, crop_y_min:crop_y_max]

    return cropped_img

plt.figure(figsize=(15, 5))

for i, img_file in enumerate(image_files):
    img = leer_digito(img_file)
    if img is not None:
        cropped_img = recortar_cuadrado(img)
        # Mostrar la imagen recortada
        # Show the cropped image
        plt.subplot(1, len(image_files), i + 1)
        plt.imshow(cropped_img, cmap="gray")
        plt.axis("off")
        plt.title(f"{img_file} \n {cropped_img.shape}")

plt.tight_layout()
plt.show()

def reescalar_8x8(img):
    """
    Reescala una imagen a 8x8 píxeles.
    Rescale an image to 8x8 pixels.

    Args:
        img (np.ndarray): La imagen de entrada como un arreglo de NumPy.
        img (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: La imagen reescalada a 8x8 píxeles como un arreglo de NumPy.
        np.ndarray: The image rescaled to 8x8 pixels as a NumPy array.
    """
    # Usamos la función resize de scikit-image para reescalar
    # We use the resize function from scikit-image to rescale
    # Se especifica la forma de destino (8, 8)
    # The target shape (8, 8) is specified
    resized_img = resize(img, (8, 8), anti_aliasing=True)
    # Opcional: Si quieres que los valores sigan en el rango 0-16
    # Optional: If you want the values to remain in the 0-16 range
    resized_img = resized_img * 16
    # Opcional: Convertir a enteros si es necesario, aunque para el modelo CNN puede no serlo
    # Optional: Convert to integers if necessary, although for the CNN model it may not be needed
    # resized_img = resized_img.astype(int)
    return resized_img

# Prueba con las imágenes procesadas previamente
# Test with previously processed images
plt.figure(figsize=(15, 5))  # Adjust figure size as needed

for i, img_file in enumerate(image_files):
        img = leer_digito(img_file)
        if img is not None:
                cropped_img = recortar_cuadrado(img)
                resized_img = reescalar_8x8(cropped_img)

                # Mostrar la imagen reescalada
                # Show the rescaled image
                plt.subplot(1, len(image_files), i + 1)
                plt.imshow(resized_img, cmap=plt.cm.gray_r)
                plt.axis('off')
                plt.title(f'{img_file}\nReescalado: {resized_img.shape}')

plt.tight_layout()
plt.show()

def predecir_digito(img):
    """
    Realiza la predicción de un dígito a partir de una imagen procesada.
    Make a digit prediction from a processed image.

    Args:
        img (np.ndarray): La imagen de entrada reescalada a 8x8.
        img (np.ndarray): The input image rescaled to 8x8.

    Returns:
        int: El dígito predicho.
        int: The predicted digit.
    """
    # Asegurarse de que la imagen tiene el shape correcto (8, 8, 1)
    # Ensure the image has the correct shape (8, 8, 1)
    # Agregamos una dimensión para el canal si no la tiene
    # Add a channel dimension if it doesn't have one
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    # Agregar una dimensión para el batch size (1 imagen)
    # Add a batch size dimension (1 image)
    img = np.expand_dims(img, axis=0)

    # Realizar la predicción con el modelo
    # Make the prediction with the model
    prediccion_one_hot = mi_modelo.predict(img)

    # Obtener el índice de la clase con mayor probabilidad
    # Get the index of the class with the highest probability
    digito_predicho = np.argmax(prediccion_one_hot, axis=1)[0]

    return digito_predicho

# Procesar todas las imágenes y hacer predicciones
# Process all images and make predictions
predicciones = []
plt.figure(figsize=(15, 5)) # Adjust figure size as needed

for i, img_file in enumerate(image_files):
        img = leer_digito(img_file)
        if img is not None:
                cropped_img = recortar_cuadrado(img)
                resized_img = reescalar_8x8(cropped_img)

                # Realizar la predicción
                # Make the prediction
                prediccion = predecir_digito(resized_img)
                predicciones.append((img_file, prediccion))

                # Mostrar la imagen con la predicción
                # Show the image with the prediction
                plt.subplot(1, len(image_files), i + 1)
                plt.imshow(resized_img, cmap=plt.cm.gray_r)
                plt.axis('off')
                plt.title(f'{img_file}\nPredicción: {prediccion}')

plt.tight_layout()
plt.show()

# Imprimir las predicciones
# Print the predictions
print("\nPredicciones para las imágenes cargadas:")
print("\nPredictions for the loaded images:")
for file, pred in predicciones:
        print(f"{file}: {pred}")