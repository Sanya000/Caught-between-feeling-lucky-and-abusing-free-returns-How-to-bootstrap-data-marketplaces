import numpy as np
import itertools
from scipy.special import comb


def calculate_score(combination_size, gamma, X):
    # This function calculates the score for a combination considering sub-combinations up to size X.
    # The score is a sum of the potential sub-combination sizes each raised to the power of gamma.
    # Base score added for the entire combination when X is 0.
    score = 0
    if X == 0:
        # Include a base score that considers the combination size itself
        score += float(combination_size) ** gamma
    else:
        for i in range(combination_size, combination_size - X, -1):
            sub_comb_count = comb(combination_size, i)  # Calculate the number of sub-combinations of size i
            score += sub_comb_count * (i ** gamma)  # Weight each sub-combination size by i^gamma
    return score
    
def find_best_combination(K, U, keys, keys_K, keys_U, B, R, gamma):
    max_score = float('-inf')
    best_comb_idx = -1
    best_key = None
    best_X = 0
    explores = -1
    
    # Iterate through combinations in keys_U since those need revealing
    for idx, key in enumerate(keys_U):
        combination_size = len(key)
        price = U[1, idx]  # Price is directly indexed as keys_U matches U
       
        # Calculate the score for this combination
        for X in range(0, combination_size - 1):
            score = calculate_score(combination_size, gamma, X)
            num_subcombs_X = 1
            cost_saved = 0
           
            if X == combination_size - 1:
                for sub_comb in itertools.combinations(key, 1):
                    if tuple(sub_comb) in keys_K:
                        cost_saved += 1
                        
            for i in range(combination_size - X, combination_size):
                num_subcombs_X += comb(combination_size, i)
                
            estimated_tests = num_subcombs_X - cost_saved 
            allowed_price = B - R * estimated_tests    
            
            # Check if this combination's price is within the allowed limit and if its score is the highest found
            if price <= allowed_price  and score > max_score:
                max_score = score
                best_comb_idx = idx
                best_key = key
                best_X = X
                explores = num_subcombs_X
                best_price = price

    if best_comb_idx != -1:
        return best_comb_idx, best_key, max_score, best_X, explores, best_price
    else:
        return None, None, None, None, None, None



def refine_combination(K, U, keys_K, keys_U, best_comb_idx, X):
    current_best_key = keys_U[best_comb_idx]
    best_sub_comb_accuracy = U[0, best_comb_idx]
    best_sub_comb_price = U[1, best_comb_idx]
    best_sub_comb = current_best_key

    # Only explore sub-combinations that exclude up to X datasets
    min_elements = max(1, len(current_best_key) - X)  # Minimum elements to keep in sub-combinations
    max_elements = len(current_best_key)  # We can go up to the full combination

    # Check all relevant sub-combinations within the range [min_elements, max_elements]
    for num_elements in range(min_elements, max_elements + 1):
        for sub_comb in itertools.combinations(current_best_key, num_elements):
            if sub_comb in keys_U:
                sub_comb_idx = keys_U.index(sub_comb)
                sub_comb_accuracy = U[0, sub_comb_idx]
                sub_comb_price = U[1, sub_comb_idx]
                if sub_comb_accuracy > best_sub_comb_accuracy:
                    best_sub_comb = sub_comb
                    best_sub_comb_accuracy = sub_comb_accuracy
                    best_sub_comb_price = sub_comb_price
            elif sub_comb in keys_K:
                # Check if a sub_combination is fully part of the freely revealed datasets
                if all(item in keys_K for item in sub_comb):
                    sub_comb_idx = keys_K.index(sub_comb)
                    sub_comb_accuracy = K[0, sub_comb_idx]  # Access accuracy directly from D
                    sub_comb_price = K[1, sub_comb_idx]
                    if sub_comb_accuracy > best_sub_comb_accuracy:
                        best_sub_comb = sub_comb
                        best_sub_comb_accuracy = sub_comb_accuracy
                        best_sub_comb_price = sub_comb_price

    return best_sub_comb, best_sub_comb_accuracy, best_sub_comb_price


def knapsack_y(K, U, B, R, keys, keys_K, keys_U, gamma):
    idx, best_key, max_score, best_X, explores, best_price = find_best_combination(K, U, keys, keys_K, keys_U, B, R, gamma)
    if idx is not None:
        best_sub_comb, best_sub_comb_accuracy, best_sub_comb_price = refine_combination(K, U, keys_K, keys_U, idx, best_X)
        if best_sub_comb_accuracy > D[0, 0]:  # Compare against the highest free dataset
            return best_sub_comb_accuracy, explores, best_X, best_sub_comb_price, best_key
    return K[0, 0], explores, best_X, best_price, best_key  # Return the best free dataset accuracy if no valid pay-to-reveal combination was better
    
    
        