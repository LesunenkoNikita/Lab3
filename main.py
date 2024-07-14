import numpy as np

def svd(A):
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)
    eigenvalues_ata, eigenvectors_ata = np.linalg.eigh(ATA)
    eigenvalues_aat, eigenvectors_aat = np.linalg.eigh(AAT)
    singular_values = np.sqrt(np.abs(eigenvalues_ata))[::-1]
    sorted_indexes = np.argsort(singular_values)[::-1]
    U = eigenvectors_aat[:, sorted_indexes]
    V = eigenvectors_ata[:, sorted_indexes]
    m, n = A.shape
    sigma = np.zeros((m, n))
    sigma[:n, :n] = np.diag(singular_values)

    return U, sigma, V.T


A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
U, sigma, Vt = svd(A)
print("Matrix A:")
print(A)
print("Matrix U:")
print(U)
print("Matrix Sigma:")
print(sigma)
print("Matrix V^T:")
print(Vt)


import pandas as pd

#Зчитування CSV файлу
file_path = 'ratings.csv'
df = pd.read_csv(file_path)
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)
ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k=3)
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, (x, y, z) in enumerate(zip(U[:, 0], U[:, 1], U[:, 2])):
    ax.scatter(x, y, z, color='b')
ax.set_title('Users')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, (x, y, z) in enumerate(zip(Vt[0, :], Vt[1, :], Vt[2, :])):
    ax.scatter(x, y, z, color='r')
ax.set_title('Films')
plt.legend()
plt.show()