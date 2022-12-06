import cv2
from matplotlib import pyplot as plt

cam_access = cv2.VideoCapture(1)

ret, frame = cam_access.read()
plt.imshow(frame)

cam_access.release()


# model training
# Load the training data
# X_train, y_train = load_data(...)

# # Fit the model
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# Load the test data
# X_test, y_test = load_data(...)

# Make predictions
# predictions = model.predict(X_test)
