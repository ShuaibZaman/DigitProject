import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.data_loader import load_mnist_data

def predict_digit(model_path, image_index):
    """
    Predicts the digit from the MNIST test dataset.

    Args:
        model_path (str): Path to the saved model.
        image_index (int): Index of the image to predict from the test dataset.

    Returns:
        None
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load test data
    _, _, x_test, y_test = load_mnist_data()

    # Get the image and true label
    image = x_test[image_index]
    true_label = np.argmax(y_test[image_index])

    # Make a prediction
    prediction = np.argmax(model.predict(image[np.newaxis, ...]))

    # Display the image and prediction
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Prediction: {prediction}, True Label: {true_label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    predict_digit("../models/mnist_digit_recognizer.h5", image_index=0)
