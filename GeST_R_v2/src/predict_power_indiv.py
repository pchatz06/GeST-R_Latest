import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class power_prediction:
    def __init__(self):
        self.power_array = []
        self.metric_array = []
        self.generation = 1


    def logarithmic_model(self, x, a, b):
        return a * np.log(x) + b

    def add_training_data(self,train_eu, train_pw):
        # Calculate first quartile (Q1)
        Q1 = np.percentile(train_pw, 15)

        # Calculate interquartile range (IQR)
        IQR = np.percentile(train_pw, 85) - Q1

        # Define lower bound for outlier detection
        lower_bound = Q1 - 1.5 * IQR

        # Identify outliers
        outliers = [index for index, value in enumerate(train_pw) if value < lower_bound]
        outlier_X = train_eu[outliers]
        outlier_Y = train_pw[outliers]

        X = np.array(train_eu)
        Y = np.array(train_pw)

        # Remove outliers from train_pw and train_eu
        X = np.delete(X, outliers)
        Y = np.delete(Y, outliers)

        # Fit curve to non-outliers
        params, _ = curve_fit(self.logarithmic_model, X, Y)

        X = X.tolist()
        Y = Y.tolist()

        self.power_array = copy.deepcopy(Y)
        self.metric_array = copy.deepcopy(X)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X, Y, label='Non-Outliers', color='blue')
        plt.scatter(outlier_X, outlier_Y, label='Outliers', color='red')
        plt.plot(train_eu, self.logarithmic_model(train_eu, *params), color="green",
                 label='Fitted Curve (Non-Outliers)')
        plt.xlabel('Metric')
        plt.ylabel('Power')
        plt.title('Actual Points vs Fitted Curve')
        plt.legend()
        plt.savefig(f'{self.generation}.png')
        plt.close()

        self.generation += 1


        # # Calculate first quartile (Q1)
        # Q1 = np.percentile(train_pw, 25)
        #
        # # Calculate interquartile range (IQR)
        # IQR = np.percentile(train_pw, 75) - Q1
        #
        # # Define lower bound for outlier detection
        # lower_bound = Q1 - 1.5 * IQR
        #
        # # Identify outliers
        # outliers = [index for index, value in enumerate(train_pw) if value < lower_bound]
        # outlier_X = train_eu[outliers]
        # outlier_Y = train_pw[outliers]
        #
        # # Remove outliers from train_pw
        # train_pw = [value for index, value in enumerate(train_pw) if index not in outliers]
        #
        # # Remove corresponding outliers from train_eu
        # train_eu = [value for index, value in enumerate(train_eu) if index not in outliers]
        #
        # # train_pw = sorted(train_pw, reverse=True)
        # self.power_array = copy.deepcopy(train_pw)
        # self.metric_array = copy.deepcopy(train_eu)
        #
        # X = np.array(self.metric_array)
        # Y = np.array(self.power_array)
        #
        # params, _ = curve_fit(self.logarithmic_model, X, Y)
        #
        # # Plot
        # plt.figure(figsize=(8, 6))
        # plt.scatter(X, Y, label='Non-Outliers', color='blue')
        # plt.scatter(outlier_X, outlier_Y, label='Outliers', color='red')
        # plt.plot(X, self.logarithmic_model(X, *params), color="green",
        # label='Fitted Curve (Non-Outliers)')
        # plt.xlabel('Metric')
        # plt.ylabel('Power')
        # plt.title('Actual Points vs Fitted Curve')
        # plt.legend()
        # plt.savefig(f'{self.generation}.png')
        # plt.close()
        #
        # self.generation += 1

    def predict_power(self, eu_to_predict):

        X = np.array(self.metric_array)
        Y = np.array(self.power_array)

        # Fit the logarithmic model to the training data
        params, _ = curve_fit(self.logarithmic_model, X, Y)

        # Use the fitted parameters to predict pw for the given eu_to_predict
        predicted_pw = self.logarithmic_model(eu_to_predict, *params)

        return predicted_pw

    # # Example usage:
    # train_eu = np.array([0.4329102418701315, 0.4318021648129907, 0.4159300704071063, 0.412975479455453, 0.40703233196892397, 0.40667646736471696, 0.3996526109129017, 0.39910036604442345, 0.3910878548444471, 0.38753245458333263])
    # train_pw = np.array([46.7794, 41.0221, 47.1759, 47.2788, 46.7852, 44.4069, 46.1393, 41.9162, 46.4127, 44.7802])
    #
    # eu_to_predict = 0.4
    #
    # predicted_pw = predict_power(train_eu, train_pw, eu_to_predict)
    # print(f"Predicted pw for eu_to_predict {eu_to_predict}: {predicted_pw}")
