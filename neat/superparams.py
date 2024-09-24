from .activation import relu, sigmoid

# distance
distance_threshold = 3.0
c1 = 1.0
c2 = 1.0
c3 = 0.4

# rate
mutation_rate = 0.8
crossover_rate = 0.7
add_edge_rate = 0.2
add_node_rate = 0.1
change_weight_rate = 0.8
disable_weight_rate = 0.1

# activation
activation_functions = sigmoid

# population
population_size = 100
species_best_size = 20

# play
action_threshold = 0.6

# others
checkpoint_path = './checkpoint'
checkpoint_rate = 5
