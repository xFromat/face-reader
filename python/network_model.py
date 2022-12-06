import tensorflow as tf

# Set up the model
model = tf.keras.Sequential()

# Add the input layer
model.add(tf.keras.layers.InputLayer(input_shape=(100, 100, 3)))

# Add the first convolutional layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

# Add the second convolutional layer
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

# Add max pooling layer
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# Add dropout layer
model.add(tf.keras.layers.Dropout(rate=0.25))

# Add flatten layer
model.add(tf.keras.layers.Flatten())

# Add dense layer
model.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Add output layer
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
