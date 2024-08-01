import numpy as np
def greedy(K,U,B,R):
    K = K[:, np.argsort(K[0])[::-1]]
    U = U[:, np.argsort(U[1])]
    best_dataset = K[:,0]
    best_accuracy = best_dataset[0]
    reserved_price = best_dataset[1]
    datasets_explored = 0
    
    for i in range(U.shape[1]):
        if B - reserved_price >= R and U[1,i] + R <= B:
            datasets_explored += 1
            B -= R
            if U[0,i] > best_accuracy:
                best_dataset = U[:,i]
                best_accuracy = best_dataset[0]
                reserved_price = best_dataset[1]

    return best_accuracy, datasets_explored
    

def inv_greedy(K,U,B,R):
    K = K[:, np.argsort(K[0])[::-1]]
    U = U[:, np.argsort(U[1])[::-1]]
    best_dataset = K[:,0]
    best_accuracy = best_dataset[0]
    reserved_price = best_dataset[1]
    datasets_explored = 0


    for i in range(U.shape[1]):
        if B - reserved_price >= R and U[1,i] + R <= B:
            datasets_explored += 1
            B -= R
            if U[0,i] > best_accuracy:
                best_dataset = U[:,i]
                best_accuracy = best_dataset[0]
                reserved_price = best_dataset[1]
                
        
    return best_accuracy, datasets_explored


def blind_buyer(K,U,B,R):
    K = K[:, np.argsort(K[0])[::-1]]
    U = U[:, np.argsort(U[1])]
    best_dataset = K[:,0]
    best_accuracy = best_dataset[0]
    reserved_price = best_dataset[1]
    num_explorations = 0
    datasets_explored = np.zeros(U.shape[1], dtype=bool)

    while B - reserved_price  >= R  and np.any((U[1,:] < (B - R - reserved_price)) & (datasets_explored == 0)): 
        unexplored_indices = np.where(datasets_explored == False)[0]
        affordable_unexplored = unexplored_indices[U[1, unexplored_indices] <= B - R - reserved_price]
        chosen_index = np.random.choice(affordable_unexplored)
        B -= R
        num_explorations += 1
        datasets_explored[chosen_index] = True

        if U[0, chosen_index] > best_accuracy:
            best_accuracy = U[0, chosen_index]
            reserved_price = U[1, chosen_index]
            
    return best_accuracy, num_explorations

