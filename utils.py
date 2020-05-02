import shutil
import os
import numpy as np
import pandas as pd
from string import ascii_lowercase, ascii_uppercase
import itertools
from tensorflow.keras import backend as K
import tensorflow as tf
import IPython.display as display

TRAIN_FILEPATH = 'pairsDevTrain.txt'
TEST_FILEPATH = 'pairsDevTest.txt'
SOURCE_PATH = 'lfw2/lfw2'
TRAIN_DST_PATH = 'data/train/'
TEST_DST_PATH = 'data/test/'
UNUSED_DST_PATH = 'data/etc/'


# -----------------------------------------------------------
#                  PREPROCESSING UTILS
# -----------------------------------------------------------

def _create_pairs(textfile_path):
    """
    This function gets the textfile_path and extracts from it the pairs of the names that are in the textfile_path.
    We use it for pairsDevText.txt nd pairsDevTrain.txt
    :param textfile_path: the path of the pairs
    :return:
    list of all the names that are in the textfile_path given.
    """
    names = set()
    with open(textfile_path) as textfile_f:
        lines = textfile_f.readlines()[1:]
    for line in lines:
        line = line[:-1].split('\t')
        if len(line) == 4:  # means we have 2 names in this line
            names.add(line[0])
            names.add(line[2])
        elif len(line) == 3:
            names.add(line[0])
    return list(names)


def _flatten_dirs(src, verbose=False):
    """
    This functions flatten a given folder so the files won't be in the inner directory.
    :param src: the path that needs to be flattened.
    :param verbose: True if we want logging of the moving, else False.
    """
    for dir in os.listdir(src):
        for file in os.listdir(os.path.join(src, dir)):
            if verbose:
                print("Moving file: " + file + "....")
            shutil.move(os.path.join(src, dir, file), os.path.join(src, file))


def _move_dirs(src_path, dest_path, folders, verbose=False):
    """
    This functions moves the dir to another place and flatten a given folder so the files won't be in the inner directory.
    :param src_path: the source path that we will move.
    :param dest_path: the destination path that we will move to from the source_path.
    :param folders: names of the folder after the moving.
    :param verbose: True if we want logging of the moving, else False.
    """
    for folder in folders:
        if verbose:
            print("Moving folder: " + folder + "....")
        shutil.move(os.path.join(src_path, folder), os.path.join(dest_path, folder))
    _flatten_dirs(src=dest_path, verbose=verbose)


def prepare_data(src_path=SOURCE_PATH, train_path=TRAIN_DST_PATH,
                 test_path=TEST_DST_PATH, unused_path=UNUSED_DST_PATH,
                 train_filepath=TRAIN_FILEPATH, test_filepath=TEST_FILEPATH, verbose=False):
    """
    This function will prepare the data in the resulted structure.
    :param src_path: the source the data is located after the download (default - 'lfw2/lfw2')
    :param train_path: the path this function will move the training examples (default - '/data/train')
    :param test_path:  the path this function will move the test examples (default - '/data/test')
    :param unusued path: the path for unused examples (default - '/data/etc/')
    :param train_filepath: the filepath of the file for creating the pairs of the name for training examples.
    :param test_filepath:the filepath of the file for creating the pairs of the name for test examples.
    :param verbose: True if we want logging of the moving, else False.
    """
    train_names = _create_pairs(train_filepath)
    test_names = _create_pairs(test_filepath)
    _move_dirs(src_path, train_path, train_names, verbose=verbose)
    _move_dirs(src_path, test_path, test_names, verbose=verbose)
    _flatten_dirs(src_path, verbose=verbose)
    shutil.move(src_path, unused_path)


def print_data_statistics(label_paths, data_name='train', label_name=['Same', 'Different']):
    """
    print statistics about about the labels of the given path
    :param label_paths: the path of the dataset
    :param data_name: the name of the dataset
    :param label_name: kind of labels.
    :return:
    """
    print('Examples for {} corpus structure:'.format(data_name))
    print('############################')
    print('View:')
    print(label_paths[:3])
    print('############################')
    print('Corpus size: {}'.format(len(label_paths)))
    print('Number of classes: {}'.format(len(label_name)))
    labels = np.array([label for _, _, label in label_paths])
    label_row_list = []
    for index, name in enumerate(label_name):
        name = name
        size = len(labels[labels == index]) / len(label_paths)
        percentage = f'{size * 100:.2f}%'
        label_row_list.append((name, len(labels[labels == index]), len(label_paths), percentage))
    df = pd.DataFrame(label_row_list, columns=['Class', 'Number in Class', 'Total', 'Percentage'])
    print(df)


def get_image_name(path):
    return path.split('/')[-1].split('.')[0]


def get_histogram_of_letters(people_names, num_letters=1):
    combination = list(ascii_uppercase)
    lower = ascii_lowercase
    if num_letters > 1:
        for i in range(1, num_letters):
            combination = [x + y for x, y in itertools.product(combination, lower)]
    hist = dict()
    for elem in combination:
        hist[elem] = sum(p.startswith(elem) for p in people_names)
    return hist


def subset_sum(numbers, desired_length, target, delta, partial=[], result=[]):
    s = sum(partial)

    # check if the partial sum is equals to target
    if s in range(target - delta, target + delta) and len(partial) == desired_length:
        result.append(partial)
    if s >= target + delta:
        return  # if we reach the number why bother to continue

    for i in range(len(numbers)):
        n = numbers[i]
        remaining = numbers[i + 1:]
        subset_sum(remaining, desired_length, target, delta, partial + [n])
    return result


def split_data_by_letters(labels, letters):
    """
    :param labels:
    :param letters:
    :return:
    """
    train = []
    val = []
    for paths in labels:
        left = get_image_name(paths[0])
        train_append = True
        for letter in letters:
            if left.startswith(letter):
                train_append = False
                break;
        if train_append:
            train.append(paths)
        else:
            val.append(paths)
    return train, val


# -----------------------------------------------------------
#                  DISTANCE UTILS
# -----------------------------------------------------------


def abs_distance(tensors, K=K):
    """
    calculates the absolute distance between two tensor
    :param tensors: 2 tensors
    :return: the absolute distance
    """
    return K.abs(tensors[0] - tensors[1])


def l2_distance(tensors, K=K):
    """
    calculates euclidiean distance
    :param tensors: 2 tensors
    :return: the euclidean distance
    """
    return K.sqrt(K.sum(K.square(abs_distance(tensors)), axis=1, keepdims=True))


def l2_distance_shape(shapes):
    """ gets tuple of shape of vectors and return the shape of the tensor"""
    shape1, shape2 = shapes
    return shape1[0], 1


def distance_acc(y_true, y_pred, th=0.5):
    """
    calculates the accuracy of the classification with threshold on distances
    :param y_true: the expected labels
    :param y_pred: the predicted labels
    :param th: th of the distance to be classified as 0 or 1
    :return:
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < th, y_true.dtype)))


# -----------------------------------------------------------
#                  PREPROCESSING UTILS
# -----------------------------------------------------------


def image2tensor(img_raw, norm=None, _resize=None):
    """
    preprocessing raw image to tensor (using libraries from tensorlow)
    :param img_raw: the raw image
    :param norm: value to normalize the image (like 255.0)
    :param resize: the shape we need to be resize to the image
    :return: the image after the preprocessing as a tensor
    """

    img_tensor = tf.image.decode_jpeg(img_raw, channels=1)
    if _resize is not None:
        img_tensor = tf.image.resize(img_tensor, _resize)
    if norm is not None:
        img_tensor = tf.cast(img_tensor, tf.float32) / norm
    return img_tensor


def load_image_as_tensor(img_path, norm=None, _resize=None):
    """
    load image and then preprocess it to tensor
    :param img_path: the path of the image
    :param norm: value to normalize the image (like 255.0)
    :param _resize: the shape we need to be resize to the image
    :return: the image after the preprocessing as a tensor
    """
    img_raw = tf.io.read_file(img_path)
    return image2tensor(img_raw, norm=norm, _resize=_resize)


def load_images_as_tensor_with_label(paths, label, norm=None, _resize=None):
    """
    load left and right image (with their label) as a tensor
    :param paths: the path of the left and right imaages
    :param label: the label of the image (y-true)
    :param norm: value to normalize the image (like 255.0)
    :param _resize: the shape we need to be resize to the image
    :return: 3-tuple of left image, right image and their label
    """
    left_img = load_image_as_tensor(paths[0], norm=norm, _resize=_resize)
    right_img = load_image_as_tensor(paths[1], norm=norm, _resize=_resize)
    return left_img, right_img, label


# -----------------------------------------------------------
#                  CREATING TENSOR DATASET UTILS
# -----------------------------------------------------------

def create_tensor_dataset(img_paths=None, labels=None, norm=None, _resize=None):
    """
    create the tensor data (full dataset, image dataset or only label dataset) to the form of tensor
    :param img_paths: the paths of the images
    :param labels: the labels of the images
    :param norm: value to normalize the image (like 255.0)
    :param _resize: the shape we need to be resize to the image
    :return: the tensor dataset
    """
    tensor_data = None
    if img_paths is not None and labels is not None:
        tensor_data = tf.data.Dataset.from_tensor_slices((img_paths, tf.cast(labels, tf.bool)))
        tensor_data = tensor_data.map(lambda paths, label: load_images_as_tensor_with_label(paths, label, norm=norm,
                                                                                            _resize=_resize),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif img_paths is not None:  # image dataset
        tensor_data = tf.data.Dataset.from_tensor_slices(img_paths)
        tensor_data = tensor_data.map(lambda paths: load_image_as_tensor(paths, norm=norm, _resize=_resize),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif labels is not None:  # label dataset
        tensor_data = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.bool))
    return tensor_data


def prepare_tensor_dataset(tensor_data, batch_size, buffer_size):
    """
    preparing the tensor so that the shuffle buffer size as large as the dataset to ensure that the data is completely
    shuffled
    :param tensor_data: the dataset
    :param batch_size:  the size of the batch
    :param buffer_size: the size of the buffer
    :return: the dataset after prefetching and shuffle
    """
    tensor_data = tensor_data.shuffle(buffer_size=buffer_size)
    tensor_data = tensor_data.repeat()
    tensor_data = tensor_data.batch(batch_size=batch_size)
    tensor_data = tensor_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return tensor_data


def init_tensor_data(tensor_data, images_labels_path, norm=255.0, _resize=[256, 256], batch_size=1, buffer_size=None,
                     verbose=True):
    """
    initializing the tensor dataset
    :param tensor_data: the dataset
    :param images_labels_path: the images and labels path
    :param norm: value to normalize the image (like 255.0)
    :param _resize: the shape we need to be resize to the image
    :param batch_size: the batch size of the tensor
    :param buffer_size: the buffer size for shuffling
    :param verbose:
    :return:
    """
    if tensor_data is None:
        if buffer_size is None:
            buffer_size = len(images_labels_path)
        images_paths = [(p[0], p[1]) for p in images_labels_path]
        labels = [ p[2] for p in images_labels_path]
        tensor_data = create_tensor_dataset(img_paths=images_paths, labels=labels, norm=norm, _resize=_resize)
        tensor_data = prepare_tensor_dataset(tensor_data, batch_size=batch_size, buffer_size=buffer_size)

        if verbose:
            print(tensor_data)

    return tensor_data


def split_train_test_tensor(tensor_data, data_len, test_size=0.2):
    """
    split the tensor to train and test tensor
    :param tensor_data:
    :param data_len:
    :param test_size:
    :return:
    """
    test_data_len = int(test_size * data_len)
    test_data = tensor_data.take(test_data_len)
    train_data = tensor_data.skip(test_data_len)
    return train_data, test_data

