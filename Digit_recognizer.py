# question_3_digit_recognizer.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def build_cnn_model():
    """
    Builds and returns a Convolutional Neural Network model for MNIST classification.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            # Convolutional layers to extract features from images
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Flatten the feature maps to feed into dense layers
            layers.Flatten(),
            # Dropout layer to prevent overfitting
            layers.Dropout(0.5),
            # Output layer with 10 units for digits 0-9
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model

# --- Main Execution ---

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Build the CNN model
model = build_cnn_model()

# Configure the model for training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
print("\n--- Starting Model Training ---")
batch_size = 128
epochs = 5
model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)

# Evaluate the final model performance
print("\n--- Evaluating Model Performance ---")
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# --- Predict and Plot Results ---
print("\n--- Displaying Predictions on Random Samples ---")

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Select 15 random images to display
num_images_to_show = 15
random_indices = np.random.choice(len(x_test), num_images_to_show, replace=False)

plt.figure(figsize=(10, 6))
plt.suptitle("Model Predictions on Random Test Samples", fontsize=16)

for i, index in enumerate(random_indices):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[index].squeeze(), cmap="gray_r")
    plt.axis('off')
    
    true_label = y_test[index]
    pred_label = predicted_labels[index]
    
    title_color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()