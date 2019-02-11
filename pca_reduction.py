# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from sklearn.decomposition import PCA

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'  # Change me to store data elsewhere


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join(data_root, filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images, max_num_images=-1):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        if num_images >= max_num_images > 0:
            break
        image_file = os.path.join(folder, image)
        try:
            image_data = (imageio.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if max_num_images > 0 and num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


num_input = 200
def whiten(pca, dataset):
    result = np.ndarray(shape=(dataset.shape[0], num_input),
                        dtype=np.float32)
    for i in range(0, dataset.shape[0]):
        transformed_vec = np.dot(dataset[i, :] - pca.mean_, pca.components_.T)
        result[i, :] = transformed_vec
    return result


def fit_pca(data_folders):
    dataset = []
    for folder in data_folders:
        print('Fitting PCA')
        dataset_reshaped = np.reshape(load_letter(folder, -1, 2000), (-1, image_size * image_size))
        for i in range(0, dataset_reshaped.shape[0]):
            dataset.append(dataset_reshaped[i])

    np.random.permutation(dataset)
    pca = PCA(n_components=num_input)
    print(np.array(dataset).shape)
    pca.fit(dataset)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.grid()
    plt.show()
    return pca

def maybe_pickle(data_folders, min_num_images_per_class, pca, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            dataset_reshaped = np.reshape(dataset, (-1, image_size * image_size))
            dataset_transformed = whiten(pca, np.array(dataset_reshaped))
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset_transformed, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names

pca = fit_pca(train_folders)
#pca = None
train_datasets = maybe_pickle(train_folders, 45000, pca)
test_datasets = maybe_pickle(test_folders, 1800, pca)


def make_arrays(nb_rows, size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, size), dtype=np.float32)
        labels = np.ndarray((nb_rows, num_classes), dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, num_input)
    train_dataset, train_labels = make_arrays(train_size, num_input)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :]
                    valid_dataset[start_v:end_v, :] = valid_letter
                    valid_labels[start_v:end_v] = np.zeros(num_classes)
                    valid_labels[start_v:end_v, label] = 1.0
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :]
                train_dataset[start_t:end_t, :] = train_letter
                train_labels[start_t:end_t] = np.zeros(num_classes)
                train_labels[start_t:end_t, label] = 1.0
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise


statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_classes = 10 # MNIST total classes (0-9 digits)

# Parameters
learning_rate = 0.001
num_steps = 2000

batch_size = 10000
display_step = 10

# tf Graph input
X = tf.placeholder("float", [None, num_input], name="X")
Y = tf.placeholder("float", [None, num_classes], name="labels")

# Store layers weight & bias
with tf.name_scope("weights"):
    weights = {
        'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1]), name="1"),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]), name="2"),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_classes]))
    }

with tf.name_scope("biases"):
    biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1]), name="1"),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2]), name="2"),
        'out': tf.Variable(tf.truncated_normal([num_classes]))
    }


# Simple neural net
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1'])), keep_prob=0.5)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = neural_net(X)

# Define loss and optimizer
with tf.name_scope("xent"):
    beta = 0
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y)) + beta * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']))
    tf.summary.scalar("loss", loss_op)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

all_vars = tf.global_variables()

def get_var(name):
    for i in range(len(all_vars)):
        if all_vars[i].name.startswith(name):
            return all_vars[i]
    return None

def print_images(imgs):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 2
    for i in range(1, columns * rows + 1):
        img = imgs[:, :, i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    i = 0
    count = 0
    train_writer = tf.summary.FileWriter('./logs/PCA/train' + str(num_input), sess.graph)
    test_writer = tf.summary.FileWriter('./logs/PCA/test' + str(num_input), sess.graph)

    num_batches = train_dataset.shape[0] // batch_size
    for step in range(1, num_steps+1):
        merge = tf.summary.merge_all()

        batch_x = train_dataset[i * batch_size:(i + 1) * batch_size, :]
        batch_y = train_labels[i * batch_size:(i + 1) * batch_size, :]

        # Run optimization op (backprop)
        train_summary, _ = sess.run([merge, train_op], feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            a, valid_summary = sess.run([accuracy, merge], feed_dict={X: valid_dataset,
                                                    Y: valid_labels})
            test_writer.add_summary(valid_summary, count)
            train_writer.add_summary(train_summary, count)

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            print("Validation Accuracy:", a)

        i += 1
        count += 1
        i = i % num_batches

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    a, pred_labels, log = sess.run([accuracy, tf.nn.softmax(
    logits=logits), logits], feed_dict={X: test_dataset,
                                        Y: test_labels})
    print("Testing Accuracy:", a)