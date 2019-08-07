import numpy as np

np.random.seed(1)
A_3 = np.random.rand(3, 4)  # feature =3 , m_training = 4
print(A_3)
keep_prob = 0.8

d3 = np.random.rand(A_3.shape[0], A_3.shape[1]) < keep_prob
print(d3)

A_3_dropout = np.multiply(A_3, d3)
print(A_3_dropout)   # A_3_dropout.shape = (  3, 4)

n_4 = 1
W_4 = np.random.rand(n_4, A_3.shape[0])
b_4 = np.random.rand(n_4, 1)
Z_4 = np.dot(W_4, A_3) + b_4
print('\n   without dropout ')
print(Z_4)
print(np.sum(Z_4))
print(np.mean(Z_4))   # * (n_4 * A_3_dropout.shape[1]

A_3_dropout /= keep_prob
Z_4 = np.dot(W_4, A_3_dropout) + b_4
print('\n   with dropout with dividing keep_prob')
print(A_3_dropout)
print(Z_4)
print(np.sum(Z_4))
print(np.mean(Z_4))   # * (n_4 * A_3_dropout.shape[1]

print(np.sum(A_3))
print(np.sum(A_3_dropout))
print(np.sum(A_3_dropout/keep_prob))
