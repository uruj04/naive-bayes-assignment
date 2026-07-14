# Naive Bayes Classification on the Iris Dataset

## Description

This project implements and compares two Naive Bayes classification algorithms—**Gaussian Naive Bayes** and **Bernoulli Naive Bayes**—using the famous Iris dataset. The models are trained and evaluated using the Scikit-learn library, and their training and testing accuracies are compared.

---

## Features

- Load and preprocess the Iris dataset
- Split the dataset into training and testing sets
- Train a Gaussian Naive Bayes classifier
- Convert continuous features into binary values using Binarizer
- Train a Bernoulli Naive Bayes classifier
- Evaluate both models using accuracy scores
- Compare the performance of GaussianNB and BernoulliNB

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## Libraries

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score
```

---

## Dataset

The project uses the **Iris dataset**, a well-known machine learning dataset containing 150 samples of iris flowers belonging to three different species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Each sample contains four numerical features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## Workflow

1. Load the Iris dataset.
2. Split the dataset into training and testing sets.
3. Train a Gaussian Naive Bayes classifier.
4. Calculate training and testing accuracy.
5. Convert numerical features into binary values using a Binarizer.
6. Train a Bernoulli Naive Bayes classifier.
7. Compare the performance of both models.

---

## Output

The notebook displays:

- Gaussian Naive Bayes Training Accuracy
- Gaussian Naive Bayes Testing Accuracy
- Bernoulli Naive Bayes Training Accuracy
- Bernoulli Naive Bayes Testing Accuracy

---

## Learning Outcomes

- Understand the Naive Bayes classification algorithm.
- Learn the difference between GaussianNB and BernoulliNB.
- Perform data preprocessing using feature binarization.
- Evaluate machine learning models using accuracy scores.
- Compare different Naive Bayes classifiers on the same dataset.

---

## Author

**Mohammad Uruj Faizan**

B.Tech Artificial Intelligence Student
