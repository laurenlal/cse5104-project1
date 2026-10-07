## TO-DO: Build grad_desc algorithm to implement into main.ipynb
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#main gradient descent algorithm. returns weights, bias, and the loss 
def gradient_descent(X, y, learning_rate=0.05, max_iterations=10000, tolerance=0.001):
    n, p = X.shape
    weights = np.zeros(p)
    bias = 0.0
    loss = []

    for i in range(max_iterations):
        y_pred = X @ weights + bias
        errors = y_pred - y

        #calculate mse 
        mse = np.mean(errors ** 2)
        loss.append(mse)

        # if i % 100 == 0:
        #     print(f"Iteration {i}, Loss: {mse}")

        if not np.isfinite(mse):
            break

        if i > 0 and abs(loss[-2] - loss[-1]) < tolerance:
            break
        

        #calc gradients
        delta_w = (2 / n) * (X.T @ errors)
        delta_b = (2 / n) * np.sum(errors)

        #update
        weights -= learning_rate * delta_w
        bias -= learning_rate * delta_b

    return weights, bias, loss

#use this function after running gradient descent to find the mse and r^2 value
def evaluate(X, y, weights, bias):
    y_pred = X @ weights + bias
    mse = np.mean((y-y_pred) ** 2)
    var = np.var(y)
    r2 = 1 - (mse / var)
    return mse, r2
