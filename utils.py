from string import ascii_lowercase, ascii_uppercase
import itertools
import os
import shutil
import pandas as pd
import numpy as np

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

# CHECK IF NEED IT
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
#
#     # -----------------------------------------------------------
#     #                  PREPROCESSING UTILS
#     # -----------------------------------------------------------
#
# def _create_pairs(textfile_path):
#     """
#     This function gets the textfile_path and extracts from it the pairs of the names that are in the textfile_path.
#     We use it for pairsDevText.txt nd pairsDevTrain.txt
#     :param textfile_path: the path of the pairs
#     :return:
#     list of all the names that are in the textfile_path given.
#     """
#     names = set()
#     with open(textfile_path) as textfile_f:
#         lines = textfile_f.readlines()[1:]
#     for line in lines:
#         line = line[:-1].split('\t')
#         if len(line) == 4:  # means we have 2 names in this line
#             names.add(line[0])
#             names.add(line[2])
#         elif len(line) == 3:
#             names.add(line[0])
#     return list(names)
#
# def _flatten_dirs(src, verbose=False):
#     """
#    This functions flatten a given folder so the files won't be in the inner directory.
#     :param src: the path that needs to be flattened.
#     :param verbose: True if we want logging of the moving, else False.
#     """
#     for dir in os.listdir(src):
#         for file in os.listdir(os.path.join(src, dir)):
#             if verbose:
#                 print("Moving file: " + file + "....")
#             shutil.move(os.path.join(src, dir, file), os.path.join(src, file))
#
# def _move_dirs(src_path, dest_path, folders, verbose=False):
#         """
#         This functions moves the dir to another place and flatten a given folder so the files won't be in the inner directory.
#         :param src_path: the source path that we will move.
#         :param dest_path: the destination path that we will move to from the source_path.
#         :param folders: names of the folder after the moving.
#         :param verbose: True if we want logging of the moving, else False.
#         """
#         for folder in folders:
#             if verbose:
#                 print("Moving folder: " + folder + "....")
#             shutil.move(os.path.join(src_path, folder), os.path.join(dest_path, folder))
#         _flatten_dirs(src=dest_path)
#
# def prepare_data(src_path, train_path,test_path, unused_path, train_filepath, test_filepath, verbose=False):
#         """
#         This function will prepare the data in the resulted structure.
#         :param src_path: the source the data is located after the download (default - 'lfw2/lfw2')
#         :param train_path: the path this function will move the training examples (default - '/data/train')
#         :param test_path:  the path this function will move the test examples (default - '/data/test')
#         :param unusued path: the path for unused examples (default - '/data/etc/')
#         :param train_filepath: the filepath of the file for creating the pairs of the name for training examples.
#         :param test_filepath:the filepath of the file for creating the pairs of the name for test examples.
#         :param verbose: True if we want logging of the moving, else False.
#         """
#         train_names = _create_pairs(train_filepath)
#         test_names = _create_pairs(test_filepath)
#         _move_dirs(src_path, train_path, train_names,verbose)
#         _move_dirs(src_path, test_path, test_names,verbose)
#         _flatten_dirs(src_path,verbose)
#         shutil.move(src_path, unused_path)
#
# def print_data_statistics(label_paths, data_name='train', label_name=['Same', 'Different']):
#         """
#         print statistics about about the labels of the given path
#         :param label_paths: the path of the dataset
#         :param data_name: the name of the dataset
#         :param label_name: kind of labels.
#         :return:
#         """
#         print('Examples for {} corpus structure:'.format(data_name))
#         print('############################')
#         print('View:')
#         print(label_paths[:3])
#         print('############################')
#         print('Corpus size: {}'.format(len(label_paths)))
#         print('Number of classes: {}'.format(len(label_name)))
#         labels = np.array([label for _, _, label in label_paths])
#         label_row_list = []
#         for index, name in enumerate(label_name):
#             name = name
#             size = len(labels[labels == index]) / len(label_paths)
#             percentage = f'{size * 100:.2f}%'
#             label_row_list.append((name, len(labels[labels == index]), len(label_paths), percentage))
#         df = pd.DataFrame(label_row_list, columns=['Class', 'Number in Class', 'Total', 'Percentage'])
#         print(df)
