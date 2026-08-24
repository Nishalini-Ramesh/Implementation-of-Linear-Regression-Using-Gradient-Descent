# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the 50_Startups.csv dataset.
2.Select R&D Spend as input x and Profit as output y, then scale x.
3.Initialize weight, bias, learning rate, and number of epochs.
4.Apply Gradient Descent to calculate predictions, loss, and update w and b.
5.Plot the loss and regression line, then print the final weight and bias.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Nishalini R
RegisterNumber:  212224040222
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ex3.csv")

x = data["R&D Spend"].values
y = data["Profit"].values


x = (x - np.mean(x)) / np.std(x)


w = 0.0          # weight
b = 0.0          # bias
alpha = 0.01     # learning rate
epochs = 100
n = len(x)

losses = []

for i in range(epochs):
    # Prediction
    y_hat = w * x + b

    # Loss (MSE)
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    # Update parameters
    w = w - alpha * dw
    b = b - alpha * db

plt.figure(figsize=(12, 5))

# Loss vs Iterations
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

# Regression Line
plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()

print("Final Weight (w):", w)
print("Final Bias (b):", b)
*/
```

## Output:
<img width="1383" height="618" alt="ex 3" src="https://github.com/user-attachments/assets/6a62712d-f2d8-4ab1-9484-fb16fd64e810" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
