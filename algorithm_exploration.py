import numpy as np
from statistics import mean
import random
import math
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal as mvn
from sklearn import preprocessing as pre
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

def find_next_best(K, U, explored_mask, current_best_price):
    # Combine explored datasets in U with all datasets in K, but only those under the current best price
    candidates = [(K[0, i], K[1, i], 'K') for i in range(K.shape[1]) if D[1, i] < current_best_price]
    candidates.extend([(U[0, i], U[1, i], 'U') for i in range(len(explored_mask)) if explored_mask[i] and U[1, i] < current_best_price])

    # Sort by accuracy, then filter by price
    if not candidates:
        return current_best_price, None, None  # No better option found
    candidates.sort(reverse=True, key=lambda x: x[0])  # sort descending by accuracy

    # Return price, dataset, and origin of the next best option
    return candidates[0][1], candidates[0][2], candidates[0][3]

def epsilon_greedy_bandit(K, U, B, R, epsilon):
    # Sort K by accuracy in descending and U by price in ascending
    K = K[:, np.argsort(K[0])[::-1]]
    U = U[:, np.argsort(U[1])]

    # Initialize exploration
    datasets_explored = np.zeros(U.shape[1], dtype=bool)
    best_accuracy = K[0, 0]
    reserved_price = K[1, 0]
    remaining_budget = B
    best_dataset = K[:, 0]

    # Loop until budget constraints are met
    while remaining_budget >= reserved_price + R:
        if random.random() < epsilon:
            # Explore: choose the cheapest unexplored dataset in U
            unexplored_indices = np.where(datasets_explored == 0)[0]
            if unexplored_indices.size > 0:
                chosen_index = unexplored_indices[0]  # Select the cheapest unexplored dataset
                datasets_explored[chosen_index] = True
                if U[0, chosen_index] > best_accuracy:
                    best_accuracy = U[0, chosen_index]
                    best_dataset = U[:, chosen_index]
                    reserved_price = U[1, chosen_index]
                remaining_budget -= R
        else:
            # Exploit: Adjust the reserved price to the next best option
            new_reserved_price, next_best_dataset, origin = find_next_best(K, U, datasets_explored, reserved_price)
            if new_reserved_price != reserved_price:
                reserved_price = new_reserved_price
                if origin == 'K':
                    best_dataset = K[:,next_best_dataset]
                else if origin == 'U':
                    best_dataset = U[:,next_best_dataset]
            remaining_budget -= reserved_price  # This line adjusts the budget based on the new reserved price

    return best_dataset[0], np.sum(datasets_explored)

