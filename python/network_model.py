from tensorflow import keras as tf
import numpy as np

def build_neural_model(nuber_of_classes: int) -> tf.Sequential:
    # Set up the model
    model = tf.Sequential([
        tf.layers.InputLayer(input_shape=(100, 100, 3)),                                                        # Input layer
        tf.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),    # First convolutional layer
        tf.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),    # Second convolutional layer
        tf.layers.MaxPool2D(pool_size=(2, 2)),                                                                  # Max pooling layer
        tf.layers.Dropout(rate=0.25),                                                                           # Dropout layer
        tf.layers.Flatten(),                                                                                    # Flatten layer
        tf.layers.Dense(units=128, activation='relu'),                                                          # Dense layer
        tf.layers.Dense(units=nuber_of_classes, activation='sigmoid'),                                          # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model: tf.Sequential, imgs_train: np.array, labels_train: np.array, imgs_val: np.array, labels_val: np.array, epochs_number: int) -> tf.Sequential:
    # # Fit the model
    model.fit(imgs_train, labels_train, validation_data = (imgs_val, labels_val) ,epochs = epochs_number, batch_size = 32, verbose = 1, use_multiprocessing = True)

    return model