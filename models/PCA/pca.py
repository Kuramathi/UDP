import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Réduire la verbosité de TensorFlow


def load_and_prepare_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32')

    X_train_flattened = X_train.reshape((X_train.shape[0], -1))
    # Standardisation
    mean = X_train_flattened.mean(axis=0)
    std = X_train_flattened.std(axis=0)
    
    epsilon = 1e-8 #evite X/0
    std += epsilon
    
    X_train_standardized = (X_train_flattened - mean) / std
    return X_train_flattened, X_train_standardized, y_train, mean, std

def values_and_vectors(data_standardized):
    cov_matrix = np.cov(data_standardized, rowvar=False) #covariance
    values, vectors = np.linalg.eigh(cov_matrix)  #valeurs propre vecteurs propres
    return values, vectors

def compress_and_decompress(X_train_flattened, X_train_standardized, vectors_trains, mean, std, num_examples=5):
    k = 200
    X_train_compressed = np.dot(X_train_standardized, vectors_trains[:, -k:])
    X_train_reconstructed = np.dot(X_train_compressed, vectors_trains[:, -k:].T) * std + mean
    
    plt.figure(figsize=(15, 10))
    for i in range(num_examples):
        plt.subplot(3, num_examples, i + 1)
        plt.title("Image originale")
        plt.imshow(X_train_flattened[i].reshape((28, 28)), cmap='gray')
        plt.axis('off')
        
        plt.subplot(3, num_examples, i + 1 + 2 * num_examples)
        plt.title("Image décompressée")
        plt.imshow(X_train_reconstructed[i].reshape((28, 28)), cmap='gray')
        plt.axis('off')
    plt.show()

def plot_2d_projection(X_train_standardized, vectors_trains, y_train):
    k = 2
    X_train_2d = np.dot(X_train_standardized, vectors_trains[:, -k:])
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap='viridis', alpha=0.6, s=2)
    plt.colorbar(scatter)
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.title('Projection 2D')
    plt.show()

def plot_3d_projection(X_train_standardized, vectors_trains, y_train):
    k = 3
    X_train_3d = np.dot(X_train_standardized, vectors_trains[:, -k:])
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2], c=y_train, cmap='viridis', alpha=0.6, s=2)
    plt.colorbar(scatter)
    ax.set_xlabel('Première composante principale')
    ax.set_ylabel('Deuxième composante principale')
    ax.set_zlabel('Troisième composante principale')
    plt.title('Projection 3D')
    plt.show()

def generate_new_data(X_train_standardized, vectors_trains, mean, std, num_new_samples=6):
    k = 2
    X_train_compressed = np.dot(X_train_standardized, vectors_trains[:, -k:])
    
    mean_compressed = X_train_compressed.mean(axis=0)
    std_compressed = X_train_compressed.std(axis=0)
    
    new_data_compressed = np.random.normal(loc=mean_compressed, scale=std_compressed, size=(num_new_samples, k))
    new_data = np.dot(new_data_compressed, vectors_trains[:, -k:].T) * std + mean
    
    plt.figure(figsize=(15, 10))
    for i in range(num_new_samples):
        plt.imshow(new_data[i].reshape((28, 28)), cmap='gray')
        plt.axis('off')
        plt.title(f"Nouvelles données {i+1}")

    plt.show()
    

def main():
    X_train_flattened, X_train_standardized, y_train, mean, std = load_and_prepare_data()
    _, vectors_trains = values_and_vectors(X_train_standardized)
    
    compress_and_decompress(X_train_flattened, X_train_standardized, vectors_trains, mean, std)
    plot_2d_projection(X_train_standardized, vectors_trains, y_train)
    plot_3d_projection(X_train_standardized, vectors_trains, y_train)
    generate_new_data(X_train_standardized, vectors_trains, mean, std)

if __name__ == "__main__":
    main()
