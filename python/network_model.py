import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

def build_neural_model(img_shape: tuple, nuber_of_classes: int) -> keras.Sequential:
    print(img_shape)
    # Set up the model
    model = keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=img_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(1024, activation='sigmoid'),
        layers.Dense(nuber_of_classes, activation='softmax')
    ])

    # Compile the model
    model = compile_model(model)

    return model

def compile_model(model: keras.Sequential) -> keras.Sequential:
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def train_model(model: keras.Sequential, imgs_train: np.array, labels_train: np.array, imgs_val: np.array, labels_val: np.array, epochs_number: int) -> keras.Sequential:
    # Fit the model
    learned_model = model.fit(imgs_train, labels_train, validation_data = (imgs_val, labels_val), epochs = epochs_number, verbose = 1)

    return model, learned_model

def evaluate_model(model: keras.Sequential, imgs_data, labels_data):
    return model.evaluate(imgs_data, labels_data, verbose = 0)