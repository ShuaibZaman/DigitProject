import tensorflow as tf

def load_mnist_data():
    """
    Loads the MNIST dataset and preprocesses it.

    Returns:
        tuple: Preprocessed training and testing data (x_train, y_train, x_test, y_test).
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add a channel dimension (required for CNNs)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    # Example usage
    x_train, y_train, x_test, y_test = load_mnist_data()
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}")
