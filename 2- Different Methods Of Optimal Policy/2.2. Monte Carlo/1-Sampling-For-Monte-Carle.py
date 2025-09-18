import numpy as np 


np.random.seed(1)

elements = [1.1, 2.2, 3.3]
probabilities = [0.2, 0.5, 0.3]

n_elements = [0, 0, 0]
N = 10000
E_x_Monte_Carlo = 0

for i in range(N):
    ele = np.random.choice(elements, 1, p=probabilities)[0]
    n_elements[elements.index(ele)] += 1
    E_x_Monte_Carlo += ele


p_elements = [0, 0, 0]
for i in range(len(n_elements)):
    p_elements[i] = n_elements[i] / sum(n_elements)

print(p_elements)


## Calculate Expectation Default Way
E_x_default = np.sum(np.array(elements) * np.array(probabilities))
print(E_x_default)

## Calculate Expectation Monte Carlo
E_x_Monte_Carlo = E_x_Monte_Carlo / N
print(E_x_Monte_Carlo)


