import numpy as np
import math

from numpy.linalg import lstsq
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def compute_gradients(a, b, x, y):
    """
    Compute the gradients of the loss function with respect to parameters a and b.
    """
    y_pred = a + b * np.log(x)
    error = y_pred - y
    gradient_a = 2 * np.mean(error)
    gradient_b = 2 * np.mean(error * np.log(x))
    return gradient_a, gradient_b


def compute_mse(y_true, y_pred):
    """
    Compute Mean Squared Error.
    """
    return np.mean((y_true - y_pred) ** 2)


def grid_search(x, y, a_range=(0, 30), b_range=(-10, 10), step=0.1):
    """
    Perform grid search over specified ranges of a and b.

    Args:
    x (np.array): Input data.
    y (np.array): Output data.
    a_range (tuple): Range of values for a (min, max).
    b_range (tuple): Range of values for b (min, max).
    step (float): Step size for the grid.

    Returns:
    tuple: The best values of a and b and the corresponding MSE.
    """
    best_a, best_b = 0, 0
    min_mse = np.inf

    for a in np.arange(a_range[0], a_range[1], step):
        for b in np.arange(b_range[0], b_range[1], step):
            y_pred = a + b * np.log(x)
            mse = compute_mse(y, y_pred)
            if mse < min_mse:
                min_mse = mse
                best_a, best_b = a, b

    return best_a, best_b, min_mse


def gradient_descent(x, y, learning_rate=0.01, iterations=1000):
    """
    Perform gradient descent to find the optimal values of a and b.
    """
    a, b = 0, 0  # starting with arbitrary values for a and b
    for _ in range(iterations):
        grad_a, grad_b = compute_gradients(a, b, x, y)
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
    return a, b


def referenceFeaturePrediction(X, y, min_base=1.1, max_base=10.0, step=0.1):
    best_mse = float('inf')
    best_base = None
    best_model = None

    # Iterate over bases from min_base to max_base at intervals of step
    current_base = min_base
    while current_base < max_base:
        # Apply logarithmic transformation with the given base
        # Adding a small constant to X to avoid log(0)
        X_trans = np.array([math.log(x, current_base) for x in X])

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2)

        # Fit a linear regression model
        reg = LinearRegression().fit(X_train.reshape(-1, 1), y_train)

        # Predict and calculate MSE
        y_pred = reg.predict(X_test.reshape(-1, 1))
        mse = mean_squared_error(y_test, y_pred)

        # Update the best model if current MSE is lower
        if mse < best_mse:
            best_mse = mse
            best_base = current_base
            best_model = reg

        current_base += step

    print(f"Best log base: {best_base}")
    print(f"Best MSE: {best_mse}")

    # Assuming best_model and best_base are obtained from the previous step
    X_new = np.array([np.max(X) + 5])  # New data point to predict for

    # Apply the same logarithmic transformation to the new data point
    X_new_trans = np.array([math.log(x, best_base) for x in X_new])

    # Predict using the best model
    y_pred = best_model.predict(X_new_trans.reshape(-1, 1))

    # Output the prediction
    print(f"Predicted value of y for X = {X_new[0]}: {y_pred[0]}")
    return y_pred


# def referenceFeaturePrediction(X, y):
#     log_models = [
#         lambda x: np.mean(y) / np.log(x),
#         lambda x: np.mean(y) / np.log10(x),
#         lambda x: np.mean(y) / np.log2(x),
#
#         lambda x: np.log(x),
#         lambda x: np.log10(x),
#         lambda x: np.log2(x),
#
#         # Additional regression models
#         lambda x: np.exp(-x),  # exponential decay regression
#         lambda x: np.power(1 / x, 2),  # power law decay regression
#
#     ]
#
#     log_model_names = ['Y/Log(x)', 'Y/Log10(x)', 'Y/Log2(x)', 'Log(x)', 'Log10(x)', 'Log2(x)',
#                        'exponential decay regression',
#                        'power law decay regression']
#
#     # define the number of folds for cross-validation
#     n_folds = 2
#
#     # initialize variables for tracking the best model and its error
#     best_model_idx = None
#     best_error = float('inf')
#
#     for i, model in enumerate(log_models):
#         errors = []
#         for train_index, test_index in KFold(n_folds).split(X):
#             # split the data into training and testing sets
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#
#             # transform the training and testing data using the current model
#             X_train_trans = model(X_train.reshape(-1, 1))
#             X_test_trans = model(X_test.reshape(-1, 1))
#
#             # fit a linear regression model to the transformed data
#             reg = LinearRegression().fit(X_train_trans, y_train)
#
#             # calculate the mean squared error on the testing data
#             y_pred = reg.predict(X_test_trans)
#             error = mean_squared_error(y_test, y_pred)
#             errors.append(error)
#
#         # calculate the average error for the current model
#         avg_error = np.mean(errors)
#
#         # update the best model index and its error if necessary
#         if avg_error < best_error:
#             best_model_idx = i
#             best_error = avg_error
#
#     # print the best model and its error
#     print(f"Best model: {log_model_names[best_model_idx]}")
#     print(f"Average error: {best_error}")
#
#     best_model = log_models[best_model_idx]
#
#     # transform the data using the selected model
#     X_trans = best_model(X.reshape(-1, 1))
#
#     # fit a linear regression model to the transformed data
#     reg = LinearRegression().fit(X_trans, y)
#
#     # predict new values of y for new values of X
#     print("Predicted Generation: ", int(X[0] + 50))
#     X_new = np.array([int(X[0] + 50)])
#     X_new_trans = best_model(X_new.reshape(-1, 1))
#     y_pred = reg.predict(X_new_trans)
#
#     # # print the predicted values
#     # for i in range(len(y_pred)):
#     #     print(f"Generation {X_new[i]}: {y_pred[i]}")
#
#     return y_pred


# def getReferenceFeatures(f):
#     features = [[] for _ in range(10)]
#     # Transform features
#     for i in range(len(features)):
#         for j in range(len(f)):
#             features[i].append(f[j][i])
#     '''
#     Features Array
#     0 - Fitness (Power)
#     1 - Load
#     2 - Store
#     3 - Scalar
#     4 - Scalar_LS
#     5 - vmul
#     6 - vadd
#     7 - vmax
#     8 - vsub
#     9 - vxor
#     '''
#
#     # Set fitness (Power) as the X
#     X = np.array(features[0])
#
#     reference_features = []
#     for i in range(1, len(features)):
#         Y = np.array(features[i])
#         print(X)
#         print(Y)
#
#         # Preparing the data for OLS
#         # The model Y = a + b*ln(x) can be rewritten as Y = a*X0 + b*X1 where X0 is a column of ones and X1 is ln(x)
#         X0 = np.ones_like(X)  # Column for the intercept
#         X1 = np.log10(X)  # Column for the coefficient of b
#         A = np.column_stack((X0, X1))  # Design matrix
#
#         # Applying the Ordinary Least Squares method
#         a_ols, b_ols = lstsq(A, Y, rcond=None)[0]
#         # a, b, error = grid_search(X, Y)
#         ypred = a_ols + b_ols * np.log10(np.max(X)+1)
#         mse_ols = compute_mse(Y, ypred)
#
#         print(f"Prediction:{ypred}")
#         print(f"MSE: {mse_ols}")
#         reference_features.append([ypred])
#         if reference_features[i - 1][0] < 0:
#             reference_features[i - 1][0] = 0
#
#         # plot
#         # x_range = np.linspace(int(np.min(X)), int(np.max(X)), int(np.max(X)) + 6)
#         # y_range = a_ols + b_ols * np.log(x_range)
#         # plt.scatter(X, Y, label=f'Feature {i}')
#         # plt.plot(x_range,y_range, label=f'Logarithmic Regression {i}')
#         # plt.scatter(np.max(X)+5, ypred, marker='x', color='red',label='Prediction')
#
#     # # Adding labels and legend
#     # plt.xlabel('X')
#     # plt.ylabel('Y')
#     # plt.title('Linear Regression and Prediction for Different Features')
#     # plt.legend()
#     #
#     # # Display the plot
#     # plt.grid(True)
#     # plt.show()
#
#     tot = 0
#     for i in range(len(reference_features)):
#         tot += reference_features[i][0]
#     print("Total Instructions = ", tot)
#     return reference_features

# Define multiple candidate functions
def linear_function(x, a, b):
    return a + b * x


def log_function(x, a, b):
    return a + b * np.log10(x)


def log_function_2(x, a, b):
    return a + b * np.log(x)


def log_function_3(x, a, b):
    return a + b * np.log2(x)


def power_function(x, a, b):
    return a * np.power(x, b)

#     # Set fitness (Power) as the X
#     X = np.array(features[0])
#
#     reference_features = []
#     for i in range(1, len(features)):
#         Y = np.array(features[i])
#         print(X)
#         print(Y)

def getReferenceFeatures(f):
    features = [[] for _ in range(10)]
    # Transform features
    for i in range(len(features)):
        for j in range(len(f)):
            features[i].append(f[j][i])
    '''
    Features Array
    0 - Fitness (Power)
    1 - Load
    2 - Store
    3 - Scalar
    4 - Scalar_LS
    5 - vmul
    6 - vadd
    7 - vmax
    8 - vsub
    9 - vxor
    '''

    # Assuming features is a 2D array with multiple features
    X = np.array(features[0])

    reference_features = []

    # Iterate through candidate functions
    candidate_functions = [linear_function, log_function, log_function_2, log_function_3, power_function]

    for i in range(1, len(features)):
        Y = np.array(features[i])

        print(X)
        print(Y)

        best_params = None
        best_mse = float('inf')
        best_function = None

        # Iterate through candidate functions and find the one with the lowest MSE
        for func in candidate_functions:
            try:
                params, covariance = curve_fit(func, X, Y, maxfev=10000)
                mse_fit = compute_mse(Y, func(X, *params))

                if mse_fit < best_mse:
                    best_params = params
                    best_mse = mse_fit
                    best_function = func

            except Exception as e:
                print(f"Error fitting {func.__name__}: {e}")

        # Make predictions using the best function and parameters
        X_max = np.max(X)
        ypred = best_function(X_max + 1, *best_params)

        # Handling negative predictions
        if ypred < 0:
            ypred = 0

        print(f"Best Function: {best_function.__name__}")
        print(f"Best Parameters: {best_params}")
        print(f"Prediction: {ypred}")
        print(f"MSE: {best_mse}")

        reference_features.append([ypred])

    tot = 0
    for i in range(len(reference_features)):
        tot += reference_features[i][0]
    print("Total Instructions = ", tot)

    return reference_features
