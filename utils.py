import shutil
import os

TRAIN_FILEPATH = 'pairsDevTrain.txt'
TEST_FILEPATH = 'pairsDevTest.txt'
SOURCE_PATH = 'lfw2/lfw2'
TRAIN_DST_PATH = 'data/train/'
TEST_DST_PATH = 'data/test/'
UNUSED_DST_PATH = 'data/etc/'


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
        for file in os.listdir(os.path.join(src,dir)):
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