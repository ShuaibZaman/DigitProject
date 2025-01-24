import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path, test_images, test_labels):
    """
    Evaluate the trained model on the test dataset.

    Args:
        model_path (str): Path to the saved model file.
        test_images (numpy.ndarray): Test images.
        test_labels (numpy.ndarray): Test labels.

    Returns:
        None
    """
    # Load the trained model
    print("Loading model from:", model_path)
    model = load_model(model_path)

    # Evaluate the model on the test set
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Predict classes
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

if __name__ == "__main__":
    # Load MNIST test dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess data
    x_test = x_test / 255.0  # Normalize pixel values to [0, 1]
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  # Add channel dimension
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)  # One-hot encode labels

    # Path to the saved model
    model_file_path = "../models/mnist_digit_recognizer.h5"

    # Evaluate the model
    evaluate_model(model_file_path, x_test, y_test)
