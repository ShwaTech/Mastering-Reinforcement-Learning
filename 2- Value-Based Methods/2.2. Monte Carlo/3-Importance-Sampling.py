import numpy as np

np.random.seed(1)

elements = [1, 2, 3]
probabilities_p = [0.3, 0.3, 0.4]
probabilities_q = [0.3, 0.6, 0.1]

n_elements = [0, 0, 0]
N = 10000

E_x = 0

for i in range(N):
    ele = np.random.choice(elements, 1, p=probabilities_q)[0]
    index_ele = elements.index(ele)
    ele *= (probabilities_p[index_ele] / probabilities_q[index_ele])

    E_x += ele


E_x /= N

print(E_x)
