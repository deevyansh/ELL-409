import os
import numpy as np
SEED = 6 # DO NOT CHANGE
rng = np.random.default_rng(SEED) # DO NOT CHANGE
from PIL import Image
from sklearn.metrics import classification_report

# Import additional allowed libraries below

class MNISTPreprocessor:
    def __init__(self, data_dir, config = None):
        """
        Initializes the MNISTPreprocessor.

        Parameters:
        - data_dir: str, the base directory where the MNIST images are stored.
        - config: config specific to your pre-processing methods
        """
        self.data_dir = data_dir
        self.classes = [str(i) for i in range(10)]
        self.image_paths = self._get_image_paths()

        '''
        You are free to add your own pre-processing steps.
        '''

    def _get_image_paths(self):
        """
        Collects all image file paths from the data directory for the specified classes.

        Returns:
        - image_paths: list of tuples (image_path, label)
        """
        image_paths = []
        for label in self.classes:
            class_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg'):
                    image_path = os.path.join(class_dir, fname)
                    image_paths.append((image_path, int(label)))
        return image_paths

    def batch_generator(self, batch_size=32, shuffle=True):
        """
        Generator that yields batches of flattened images and their labels.

        Parameters:
        - batch_size: int, the number of images per batch.
        - shuffle: bool, whether to shuffle the data before each epoch.

        Yields:
        - X_batch: numpy array of shape (batch_size, 784)
        - y_batch: numpy array of shape (batch_size,)
        """
        image_paths = self.image_paths.copy()
        if shuffle:
            np.random.shuffle(image_paths)

        num_samples = len(image_paths)
        for offset in range(0, num_samples, batch_size):
            batch_samples = image_paths[offset:offset+batch_size]
            X_batch = []
            y_batch = []
            for image_path, label in batch_samples:
                image = Image.open(image_path).convert('L')  # Ensure image is grayscale
                image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]
                image_vector = image_array.flatten()  # Convert to 1D vector
                X_batch.append(image_vector)
                y_batch.append(label)
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch

    def get_all_data(self):
        """
        Loads all data into memory.

        Returns:
        - X: numpy array of shape (num_samples, 784)
        - y: numpy array of shape (num_samples,)
        """
        X = []
        y = []
        for image_path, label in self.image_paths:
            image = Image.open(image_path).convert('L')
            image_array = np.array(image).astype(np.float32) / 255.0
            image_vector = image_array.flatten()
            X.append(image_vector)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        return X, y


def filter_dataset(X, y, entry_number_digit):
    target_digits = [
        entry_number_digit % 10,
        (entry_number_digit - 1) % 10,
        (entry_number_digit + 1) % 10,
        (entry_number_digit + 2) % 10
    ]

    # Use np.isin for a cleaner condition
    condition = np.isin(y, target_digits)
    relevant_pos = np.where(condition)[0]
    X = X[relevant_pos]
    y = y[relevant_pos]
    return X, y


def convert_labels_to_svm_labels(arr, svm_pos_label = 0):
    return np.where(arr == svm_pos_label, 1, -1)

def get_metrics_as_dict(preds, true):
    assert(
        preds.shape == true.shape
    ), "Shape Mismatch. Assigning 0 score"

    report = classification_report(true, preds, output_dict=True)
    return report

def val_score(preds, true):
    assert(
        preds.shape == true.shape
    ), "Shape Mismatch. Assigning 0 score"

    report = classification_report(true, preds, output_dict=True)
    return report["macro avg"]['f1-score']
