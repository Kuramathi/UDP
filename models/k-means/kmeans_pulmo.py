import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class KMeansCustom:
    def __init__(self, n_clusters, max_iter=100, tol=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.permutation(len(X))
        self.cluster_centers = X[random_indices[:self.n_clusters]]

        for i in tqdm(range(self.max_iter), desc="Training K-Means"):
            self.labels_ = self.assign_clusters(X)
            new_centers = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.n_clusters)])
            if np.all(np.abs(new_centers - self.cluster_centers) < self.tol):
                break
            self.cluster_centers = new_centers

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self.assign_clusters(X)

def load_image_data(data_dir, img_size=(28, 28)):
    labels = ['pneumonia', 'normal', 'lung_opacity', 'covid']
    data = []
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=img_size, color_mode='grayscale')
            img_array = img_to_array(img).astype('float32') / 255.0
            data.append(img_array.flatten())
    return np.array(data)

def train_kmeans(data, n_clusters):
    kmeans = KMeansCustom(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans

def compress(data, kmeans):
    compressed_data = kmeans.predict(data)
    return compressed_data

def decompress(compressed_data, kmeans):
    decompressed_data = kmeans.cluster_centers[compressed_data]
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

def generate_new_images(kmeans, n_images=10, noise_level=0.1):
    new_images = []
    for _ in range(n_images):
        cluster_idx = np.random.choice(kmeans.n_clusters)
        cluster_center = kmeans.cluster_centers[cluster_idx]
        new_image = cluster_center + np.random.normal(scale=noise_level, size=cluster_center.shape)
        new_image = np.clip(new_image, 0, 1)
        new_images.append(new_image)
    return np.array(new_images)

if __name__ == "__main__":
    data_dir = '../../data'
    data = load_image_data(data_dir)
    kmeans_model = train_kmeans(data, n_clusters=10)
    compressed_data = compress(data, kmeans=kmeans_model)
    decompressed_data = decompress(compressed_data, kmeans=kmeans_model)
    generated_images = generate_new_images(kmeans_model, n_images=10, noise_level=0.1)
    plot_images(data, title='Original Images', n_rows=1, n_cols=5)
    plot_images(decompressed_data, title='Decompressed Images', n_rows=1, n_cols=5)
    plot_images(generated_images, title='Generated Images', n_rows=1, n_cols=5)
