import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

from sklearn.decomposition import PCA

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
plt.rcParams['image.cmap'] = 'gray'

def load_letter(folder, dataset, index_start, index_count):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)

    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (imageio.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[index_start + num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

        if index_count <= num_images:
            break


    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

total_images = 50000

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
dataset = np.ndarray(shape=(total_images, image_size, image_size),
                     dtype=np.float32)
i = 0
for letter in letters:
    load_letter("notMNIST_large\\" + letter, dataset, i * total_images // 10, total_images // 10)
    i += 1

dataset = np.random.permutation(dataset)

plt.figure()
for i in range(0, 25):
    plt.subplot(5, 5, i+1)
    plt.imshow(dataset[i, :, :])
plt.show()

pca784= PCA(n_components=784)
pca200= PCA(n_components=200)
pca75= PCA(n_components=75)
pca25= PCA(n_components=25)
pca8= PCA(n_components=8)
a_reshaped = np.reshape(dataset, (-1, image_size * image_size))

pca784.fit(a_reshaped)
pca200.fit(a_reshaped)
pca75.fit(a_reshaped)
pca25.fit(a_reshaped)
pca8.fit(a_reshaped)

plt.plot(np.cumsum(pca784.explained_variance_ratio_))
plt.ylim(0.8, 1.0)
plt.grid()
plt.show()

a_PCA784 = np.dot(a_reshaped - pca784.mean_, pca784.components_.T)
a_PCA200 = np.dot(a_reshaped - pca200.mean_, pca200.components_.T)
a_PCA75 = np.dot(a_reshaped - pca75.mean_, pca75.components_.T)
a_PCA25 = np.dot(a_reshaped - pca25.mean_, pca25.components_.T)
a_PCA8 = np.dot(a_reshaped - pca8.mean_, pca8.components_.T)

def reconstruct(pca, vec):
    return np.reshape(pca.mean_ + np.dot(vec, pca.components_), (image_size,image_size))

def whiten(pca, vec):
    transformed_vec = np.dot(vec - pca.mean_, pca.components_.T)
    return (np.dot(transformed_vec / np.sqrt(pca.explained_variance_ + 0.001), pca.components_)).reshape(image_size, image_size)

def whiten2(pca, vec):
    transformed_vec = np.dot(vec, pca.components_.T)
    return (np.dot(transformed_vec / pca.singular_values_, pca.components_) * np.sqrt(a.shape[0])).reshape(image_size, image_size)

for i in range(0, 5):
    sp = plt.subplot(5,6,1 + i*6)
    plt.imshow(dataset[i, :, :])
    if i == 0:
        sp.set_title("Original")
    sp = plt.subplot(5,6,2 + i*6)
    plt.imshow(reconstruct(pca784, a_PCA784[i]))
    if i == 0:
        sp.set_title("784")
    sp = plt.subplot(5,6,3 + i*6)
    plt.imshow(reconstruct(pca200, a_PCA200[i]))
    if i == 0:
        sp.set_title("200")
    sp = plt.subplot(5, 6, 4 + i * 6)
    plt.imshow(reconstruct(pca75, a_PCA75[i]))
    if i == 0:
        sp.set_title("75")
    sp = plt.subplot(5, 6, 5 + i * 6)
    plt.imshow(reconstruct(pca25, a_PCA25[i]))
    if i == 0:
        sp.set_title("25")
    sp = plt.subplot(5, 6, 6 + i * 6)
    plt.imshow(reconstruct(pca8, a_PCA8[i]))
    if i == 0:
        sp.set_title("8")


plt.show()