# Neural Network from Scratch and Perceptron Classification using scikit-learn

## Project Overview
This project demonstrates **foundational neural network concepts** by implementing:

1. A **single-layer neural network (perceptron) from scratch** using NumPy
2. A **Perceptron classifier using scikit-learn** applied to the Iris dataset

The goal is to bridge **theoretical understanding** of neural networks with **practical machine learning implementation**, making this project ideal for learning, coursework, and portfolio demonstration.

---

## Objectives
- Understand how a basic neural network learns through weight updates
- Implement sigmoid activation and gradient-based learning manually
- Train a perceptron classifier using a real-world dataset
- Compare manual neural learning with library-based ML models
- Evaluate classification performance using accuracy metrics

---

## Technologies Used
- Python 3
- NumPy
- Pandas
- scikit-learn

---

## Part 1: Neural Network from Scratch

### Dataset
- Loaded from `pre.csv`
- Features are numerical
- Last column represents the target output
- Target values reshaped to column vector format

---

### Neural Network Architecture
- Single-layer neural network
- No hidden layers
- One output neuron
- Fully connected input layer

---

### Core Components Implemented
- **Sigmoid Activation Function**
  - Converts weighted sums into probabilities
- **Sigmoid Derivative**
  - Used for weight adjustment during learning
- **Random Weight Initialization**
  - Ensures non-zero gradient flow
- **Forward Propagation**
  - Computes predictions using dot product and activation
- **Error Calculation**
  - Difference between predicted output and true labels
- **Weight Update Rule**
  - Adjusts synaptic weights using gradient-based learning

---

### Training Process
- Model trained for **100,000 iterations**
- Weights updated iteratively to minimize prediction error
- Final trained weights printed after training
- Final output values displayed

---

### Learning Outcome
This section provides a **deep understanding of how neural networks learn internally**, without relying on machine learning libraries.

---

## Part 2: Perceptron Classification using scikit-learn

### Dataset
- **Iris Dataset** from scikit-learn
- Binary classification task:
  - Class 0 (Setosa) → 1
  - Other classes → 0

---

### Data Preparation
- Feature matrix extracted from Iris dataset
- Target labels converted into binary format
- Data split into training and testing sets (80/20 split)

---

### Model Description
#### Perceptron Classifier
- Linear binary classifier
- Learns a decision boundary using weight updates
- Suitable for linearly separable data

The model is implemented using scikit-learn’s `Perceptron` class.

---

### Model Training and Evaluation
- Model trained on training dataset
- Predictions made on test dataset
- Performance evaluated using:
  - **Accuracy Score**

---

## Results
- Neural network from scratch successfully learns optimal weights
- Perceptron classifier achieves strong accuracy on Iris dataset
- Demonstrates effectiveness of linear classifiers on separable data

---

## How to Run the Project

### Prerequisites
Install required libraries:
- numpy
- pandas
- scikit-learn

---

### Steps
1. Place `pre.csv` in the specified directory or update the file path
2. Run the Python script
3. Observe:
   - Initial and final synaptic weights
   - Neural network output
   - Perceptron accuracy score

---

## Learning Outcomes
- Clear understanding of neural network fundamentals
- Hands-on experience implementing learning algorithms from scratch
- Practical exposure to scikit-learn classifiers
- Ability to compare manual vs library-based ML approaches

---

## Future Improvements
- Add learning rate control
- Implement loss function tracking
- Extend to multi-layer neural networks
- Apply normalization for improved convergence
- Compare with Logistic Regression and SVM

---

## Use Case
This project is suitable for:
- Machine Learning and AI portfolios
- Academic coursework and labs
- Interview preparation
- Understanding neural network foundations

---

## Author
Soban Saeed
Developed as an educational project to demonstrate neural network learning from scratch and perceptron-based classification using scikit-learn.
