**Iris Classification using Machine Learning**

A Neural Network based Machine Learning project to classify Iris flower species using TensorFlow and Scikit-Learn.
The model predicts species based on sepal and petal measurements.

**BADGES**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Project Structure**

iris-classification-ml/
│
├── iris.csv
├── Iris_model.ipynb
├── iris_save.keras
├── README.md
├── LICENSE
└── .gitignore

**PROJECT OVERVIEW**

This project builds a deep learning classifier to predict:

Iris-setosa
Iris-versicolor
Iris-virginica
Using a neural network with:
2 Hidden Layers
ReLU Activation
Softmax Output Layer for 3 classes
Accuracy: ~95–98%


**MODEL ARCHITECTURE**

Input: 4 features
Dense(10, relu)
Dense(10, relu)
Dense(3, softmax)


**TECHNOLOGY USED**

Python
TensorFlow
Scikit-Learn
Pandas
NumPy

**Sample Prediction**

sample = np.array([[5.9, 3.0, 5.1, 1.8]])
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(label_encoder.inverse_transform([np.argmax(prediction)]))

**License**

MIT License
