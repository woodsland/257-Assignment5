# COMP257 - Unsupervised & Reinforcement Learning (Section 002)
# Assignment 5 - Autoencoders
# Name: Wai Lim Leung
# ID  : 301276989
# Date: 9-Nov-2023

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import KFold
from keras.layers import Dense, Flatten, Reshape, Input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Part 1 & 2 - Retrieve & Load Olivetti Faces
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
of_data = olivetti_faces.data
of_target = olivetti_faces.target
of_images = olivetti_faces.images
of_labels = olivetti_faces.target

print()
print("Part 1 & 2 - Olivetti Faces")
print("Data Shape  :", of_data.shape)
print("Target Shape:", of_target.shape)

fig = plt.figure(figsize=(6, 2))
for i in range(3):
    face_image = of_data[i].reshape(64, 64)
    position = fig.add_subplot(1, 3, i + 1)
    position.imshow(face_image, cmap='gray')
    position.set_title(f"Person {of_target[i]}")
    position.axis('off')
plt.tight_layout()
plt.show()

# Part 3 - Training, Validation & Test
X_temp, X_test, y_temp, y_test = train_test_split(of_images, of_labels, test_size=0.2, stratify=of_labels, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
print()
print("Part 3")
print("Training  :", [np.sum(y_train == i) for i in range(40)])
print("Validation:", [np.sum(y_val == i)   for i in range(40)])
print("Test      :", [np.sum(y_test == i)  for i in range(40)])

# Part 4 - PCA Preserving 99% Variance
pca = PCA(n_components=0.99, whiten=True, random_state=42)
faces_data_pca = pca.fit_transform(of_data)
print()
print("Part 4 - PCA Preserving 99% Variance")
print(faces_data_pca)

# Part 5 - Autoencoder
def build_autoencoder(input_shape, encoding_dim):
    encoding_size = encoding_dim
    decoding_size = np.prod(input_shape)

    input_img = Input(shape=(decoding_size,))
    encoded = Dense(encoding_size, activation='relu')(input_img)
    decoded = Dense(decoding_size, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

flattened_images = of_data.reshape((of_data.shape[0], -1))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_index, test_index in kf.split(flattened_images):
    X_train, X_val = flattened_images[train_index], flattened_images[test_index]
    autoencoder = build_autoencoder(input_shape=(64, 64), encoding_dim=64)
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_val, X_val))
    score = autoencoder.evaluate(X_val, X_val, verbose=0)
    scores.append(score)

average_score = np.mean(scores)
print()
print("Part 5 - kFold and Layer")
print("Average Validation Score:", average_score)

# Part 6 - Best Model
print()
print("Part 6 - Best Model")
X_test_flattened = X_test.reshape((X_test.shape[0], -1))
reconstructed_images = autoencoder.predict(X_test_flattened)
reconstructed_images = reconstructed_images.reshape(X_test.shape)

# Plot Original & Reconstructed Image
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_images[i].reshape(64, 64), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()