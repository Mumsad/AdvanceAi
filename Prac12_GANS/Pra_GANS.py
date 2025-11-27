import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST images
(x_train, _), (_, _) = mnist.load_data()

# Normalize to [0,1]
x_train = x_train / 255.0

# Pick 16 random images
idx = np.random.choice(len(x_train), 16, replace=False)
sample_imgs = x_train[idx]

# Plot 4x4 grid
plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(sample_imgs[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.savefig("mnist_grid_output.png")
plt.show()
