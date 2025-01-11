import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers as opt
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image

IMG_SHAPE = (48, 48)

def build_neural_model(nuber_of_classes: int, img_shape: tuple = IMG_SHAPE) -> keras.Sequential:
    # Set up the model
    model = keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=img_shape),
        layers.InputLayer(input_shape = img_shape),
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(factor=(-0.05, 0.05)),
        # tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor = (-0.2, 0.2), width_factor = (-0.2, 0.2)),
        layers.Conv2D(32, kernel_size=(8,8), activation='elu', kernel_initializer='he_normal', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(8,8), activation='elu', kernel_initializer='he_normal', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.5),
        layers.Conv2D(64, kernel_size=(5,5), activation='elu', kernel_initializer='he_normal', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(5,5), activation='elu', kernel_initializer='he_normal', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.4),
        layers.Conv2D(128, kernel_size=(4,4), activation='elu', kernel_initializer='he_normal', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(4,4), activation='elu', kernel_initializer='he_normal', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=(3,3), activation='elu', kernel_initializer='he_normal', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.6),
        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(nuber_of_classes, activation='softmax')
    ])

    # Compile the model
    model = compile_model(model)

    return model

def compile_model(model: keras.Sequential) -> keras.Sequential:
    custom_optimizer = opt.Nadam(learning_rate=0.35)
    model.compile(optimizer='nadam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

def train_model(model: keras.Sequential, imgs_train: np.array, labels_train: np.array, imgs_val: np.array, labels_val: np.array, epochs_number: int, classes_weights: dict = dict()) -> keras.Sequential:
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00005,
        patience=11,
        verbose=1,
        restore_best_weights=True,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.6,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    callbacks = [
        early_stopping,
        lr_scheduler,
    ]
    if len(classes_weights) > 0:
        # Fit the model
        learned_model = model.fit(imgs_train, labels_train, validation_data = (imgs_val, labels_val), epochs = epochs_number, batch_size = 128, callbacks=callbacks, class_weight=classes_weights)
    else:
        learned_model = model.fit(imgs_train, labels_train, validation_data = (imgs_val, labels_val), epochs = epochs_number, batch_size = 256, callbacks=callbacks)

    return model, learned_model.history

def evaluate_model(model: keras.Sequential, imgs_data, labels_data):
    return model.evaluate(imgs_data, labels_data, verbose = 0)

def test_photo(model: keras.Sequential, photo: np.ndarray):
    if photo is None:
        print("test_photo(): No picture detected")
        return
    photo = np.asarray(Image.fromarray(photo).resize(IMG_SHAPE))
    predicted_class = model.predict(photo.reshape([1, 48, 48]))
    print(predicted_class)
    return predicted_class.argmax()