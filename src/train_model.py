from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.data_loader import load_mnist_data

def create_model():
    """
    Defines the CNN model architecture.
    Returns: Sequential: Compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")  # 10 output classes for digits 0-9
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    # Load data
    x_train, y_train, x_test, y_test = load_mnist_data()

    # Create model
    model = create_model()
    model.summary()

    # Define early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Train the model
    model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Save the model
    model.save('../models/mnist_digit_recognizer.h5')
    print("Model saved")
