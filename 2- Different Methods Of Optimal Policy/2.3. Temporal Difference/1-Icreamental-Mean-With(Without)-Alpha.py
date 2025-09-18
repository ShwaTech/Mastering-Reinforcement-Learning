import numpy as np

np.random.seed(1)

elements = [1.1, 2.2, 3.3]
probabilities = [0.2, 0.5, 0.3]

## Calculate Expectation Default Way
E_x_default = np.sum(np.array(elements) * np.array(probabilities))
print(E_x_default)


## Calculate Expectation Monte Carlo - Without Alpha
N=10_000
E_x_MonteCarlo=0

for i in range(N):
    ele = np.random.choice(elements, 1, probabilities)[0]

    E_x_MonteCarlo = E_x_MonteCarlo + (1 / (i+1)) * (ele - E_x_MonteCarlo)

print(E_x_MonteCarlo)


## Calculate Expectation Monte Carlo - With Alpha
N=10_000
E_x_MonteCarlo=0
alpha=0.01

for i in range(N):
    ele = np.random.choice(elements, 1, probabilities)[0]

    E_x_MonteCarlo = E_x_MonteCarlo + alpha * (ele - E_x_MonteCarlo)

print(E_x_MonteCarlo)
