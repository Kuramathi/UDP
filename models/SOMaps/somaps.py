import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class SOM:
    def __init__(self, m, n, dim, n_iterations, learning_rate=0.5):
        self.m = m  # Grid height
        self.n = n  # Grid width
        self.dim = dim 
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights = np.random.rand(m, n, dim)
        self.initial_learning_rate = learning_rate

    def fit(self, X):
        for iteration in tqdm(range(self.n_iterations), desc="Training SOM"):
            for x in X:
                bmu_index = self.find_bmu(x)
                self.update_weights(x, bmu_index, iteration)

    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), (self.m, self.n))

    def update_weights(self, x, bmu_index, iteration):
        learning_rate = self.learning_rate * (1 - iteration / self.n_iterations)
        sigma = self.m / 2 * (1 - iteration / self.n_iterations)

        for i in range(self.m):
            for j in range(self.n):
                w = self.weights[i, j]
                d = np.linalg.norm(np.array([i, j]) - np.array(bmu_index))
                if d < sigma:
                    influence = np.exp(-d**2 / (2 * sigma**2))
                    self.weights[i, j] += influence * learning_rate * (x - w)

    def predict(self, X):
        return np.array([self.find_bmu(x) for x in X])

def load_mnist_data():
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    return X_train

def train_som(data, m, n, dim, n_iterations, learning_rate=0.5):
    som = SOM(m, n, dim, n_iterations, learning_rate)
    som.fit(data)
    return som

def compress(data, som):
    compressed_data = som.predict(data)
    return compressed_data

def decompress(compressed_data, som):
    decompressed_data = np.array([som.weights[i, j] for i, j in compressed_data])
    return decompressed_data

def plot_images(images, title, n_rows=10, n_cols=10):
    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
    plt.suptitle(title, fontsize=16)
    for i, image in enumerate(images[:n_rows * n_cols]):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def generate_new_images(som, n_images=10, noise_level=0.1):
    new_images = []
    for _ in range(n_images):
        i = np.random.randint(0, som.m)
        j = np.random.randint(0, som.n)
        cluster_center = som.weights[i, j]
        new_image = cluster_center + np.random.normal(scale=noise_level, size=cluster_center.shape)
        new_image = np.clip(new_image, 0, 1)
        new_images.append(new_image)
    return np.array(new_images)

if __name__ == "__main__":
    data = load_mnist_data()
    som_model = train_som(data, m=10, n=10, dim=data.shape[1], n_iterations=50) 
    compressed_data = compress(data, som=som_model)
    decompressed_data = decompress(compressed_data, som=som_model)
    generated_images = generate_new_images(som_model, n_images=10, noise_level=0.1)
    plot_images(data, title='Original Images', n_rows=1, n_cols=5)
    plot_images(decompressed_data, title='Decompressed Images', n_rows=1, n_cols=5)
    plot_images(generated_images, title='Generated Images', n_rows=1, n_cols=5)
