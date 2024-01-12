import heapq

def astar(graph, start, goal):
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0, start, []))  # Priority queue with initial cost, node, and path

    while open_set:
        current_cost, current_node, path = heapq.heappop(open_set)

        if current_node == goal:
            return path + [current_node]

        if current_node in closed_set:
            continue

        closed_set.add(current_node)

        for neighbor, cost in graph[current_node].items():
            heapq.heappush(open_set, (current_cost + cost + heuristic(neighbor, goal), neighbor, path + [current_node]))

    return None  # No path found

def heuristic(node, goal):
    # Replace this with an appropriate heuristic function (e.g., Euclidean distance for grid-based maps)
    return 0

# Example usage:
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 2},
    'C': {'A': 4, 'D': 5},
    'D': {'B': 2, 'C': 5}
}

start_node = 'A'
goal_node = 'D'

print(astar(graph, start_node, goal_node))




import pandas as pd

def candidate_elimination(data):
    # Initialize general and specific boundaries
    general_boundary = [["?" for i in range(len(data.columns) - 1)] for j in range(len(data))]
    specific_boundary = data.iloc[0, :-1].tolist()

    for i in range(len(data)):
        if data.iloc[i, -1] == "yes":
            # Generalize specific boundary
            for j in range(len(specific_boundary)):
                if specific_boundary[j] != data.iloc[i, j]:
                    specific_boundary[j] = "?"
                    general_boundary[j][j] = "?"
        else:
            # Specialize general boundary
            for j in range(len(general_boundary)):
                if general_boundary[j][j] != data.iloc[i, j]:
                    general_boundary[j] = ['?' for k in range(len(data.columns) - 1)]

    return general_boundary, specific_boundary

# Example usage
data = pd.DataFrame({
    "Sky": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy"],
    "AirTemp": ["Warm", "Warm", "Hot", "Mild", "Cool"],
    "Humidity": ["Normal", "High", "High", "Normal", "Normal"],
    "Wind": ["Weak", "Strong", "Weak", "Weak", "Strong"],
    "Water": ["Yes", "No", "Yes", "Yes", "No"]
})

general_boundary, specific_boundary = candidate_elimination(data)
print("General Boundary:", general_boundary)
print("Specific Boundary:", specific_boundary)






import numpy as np

# Data normalization
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X / np.amax(X, axis=0)  # maximum of X array longitudinally
y = y / 100

# Sigmoid Function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Variable initialization
epoch = 5000
lr = 0.1
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Weight and bias initialization
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wout = np.random.uniform(size=(hidden_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training loop
for _ in range(epoch):
    # Forward Propagation
    hlayer_act = sigmoid(np.dot(X, wh) + bh)
    output = sigmoid(np.dot(hlayer_act, wout) + bout)

    # Backpropagation
    EO = y - output
    d_output = EO * output * (1 - output)
    EH = d_output.dot(wout.T)
    d_hiddenlayer = EH * hlayer_act * (1 - hlayer_act)

    # Weight update
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr

print("Input:\n", X)
print("Actual Output:\n", y)
print("Predicted Output:\n", output)
