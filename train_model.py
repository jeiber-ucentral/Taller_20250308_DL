#------------------------------------------------------------------------------------#
#----------------- CLASIFICACION A CIFAR10 CON ARQUITECTURA ALEXNET -----------------#
#------------------------------------------------------------------------------------#

#######################################
# # # CARGUE DE LIBRERIAS Y DATOS # # # 
#######################################
# Cargue de librerias
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import recall_score, precision_score, f1_score

# Cargue de datos CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train.astype("float32") / 255.0 # Comentar para cifar10
# x_test = x_test.astype("float32") / 255.0 # Comentar para cifar10

print(x_train.shape)
print(y_train.shape)

#################################################
# # # Creacion de la arquitectura de la red # # # 
#################################################
# Modelo ALexNet : creacion de arquitectura
model = keras.Sequential([
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Flatten(),
    Dense(units=4096, activation='relu'),
    Dense(units=4096, activation='relu'),
    Dense(units=10, activation='softmax')
 ])


'''
# Para mnist
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])
'''

# Compilando el modelo
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


# Entrenando el modelo
model.fit(
    x_train, 
    y_train,
    epochs = 1, #10,
    validation_data = (x_test, y_test)
)

################################
# # # Evaluacion de la red # # # 
################################
# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test)

# Uso del modelo sobre base test
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1) # para dejar la categoria de mayor probabilidad de pertenencia

# Calcular metricas de desempeno
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Guardar los resultados en un df
results_df = pd.DataFrame({
    'Metric': ['Loss', 'Accuracy', 'Recall', 'Precision', 'F1-Score'],
    'Value': [loss, accuracy, recall, precision, f1]
})

print(results_df)

# Guardar las metricas en un csv
results_df.to_csv("resultados_AlexNet_CIFAR10.csv", index=False)

# Guardar el modelo
# model.save('AlexNet_CIFAR10_model.h5')

print("TODO EJECUTADO!")


