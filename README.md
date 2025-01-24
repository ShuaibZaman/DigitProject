This project implements a Convolutional Neural Network (CNN) model to recognize handwritten digits using the MNIST dataset. 
The application includes training, evaluating, and predicting digits, with utilities to save and load models for future use.


DigitProject/
├── models/
│   └── mnist_digit_recognizer.h5  # Saved model file
├── src/
│   ├── data_loader.py             # Loads the MNIST dataset
│   ├── train.py                   # Trains the model
│   ├── evaluate.py                # Evaluates the trained model
│   └── predict.py                 # Predicts using the trained model
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore file
