import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_cifar10(validation_size=10000):
    """
    Load the cifar10 image dataset.

    Args:
        validation_size (int) -- The number of samples to include in the
            validation dataset. Cannot exceed the number of samples in the
            training set (50,000).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test -- Training, validation,
            and testing datasets.
    """
    base_path = '../data/'
    X_test, y_test = load_images_from_dir(base_path + 'test_images/')
    X_train, y_train = load_images_from_dir(base_path + 'train_images/')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
        test_size=validation_size, stratify=y_train)
    return ((X_train, to_categorical(y_train)), (X_val, to_categorical(y_val)),
        (X_test, to_categorical(y_test)))


def load_images_from_dir(abs_path):
    """Load up the cifar10 images and labels from a directory."""
    for ix, fname in enumerate(os.listdir(abs_path)):
        with open(abs_path + fname, 'rb') as f:
            f_dict = pickle.load(f, encoding='bytes')
            f_labels = np.array(f_dict[b'labels'])
            f_data = f_dict[b'data'].reshape(
                [10000, 3, 32, 32]).transpose([0, 2, 3, 1])
        labels = np.concatenate((labels, f_labels)) if ix else f_labels
        data = np.concatenate((data, f_data)) if ix else f_data
    return data, labels
