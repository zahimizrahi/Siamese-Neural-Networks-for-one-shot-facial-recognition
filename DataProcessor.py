import os
import IPython.display as display
import glob
import numpy as np
import shutil
import pandas as pd

class DataProcessor:
    def __init__(self, src_path, train_dst_path, test_dst_path, unused_dst_path,
                 train_file_path, test_file_path, verbose=False):
        """
        Constructor of DataProcessor.
        :param data_path:
        :param label_file:
        :param print_imgs:
        """
        self.src_path = src_path
        self.train_dst_path = train_dst_path
        self.test_dst_path = test_dst_path
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.unused_path = unused_dst_path
        self.verbose = verbose
        self.label_paths = []

    def load_data(self, count_print_imgs=3):
        """
        before we begin the training, we need to load the dataset and prepare it.
        we use it: https://www.tensorflow.org/tutorials/load_data/images
        :return:
        """
        train_test_list_of_paths = [[] , []]
        for index,data_path in enumerate([self.train_dst_path, self.test_dst_path]):
            full_path = os.path.abspath(data_path)
            data_paths = list(glob.glob(f'{full_path}/*'))
            data_paths = [str(path) for path in data_paths if os.path.isfile(path)]
            if self.verbose:
                print("{} Image paths were loaded!".format(len(data_paths)))

                for n in range(count_print_imgs):
                    print('#########################')
                    print('Example:')
                    image_path = np.random.choice(data_paths)
                    print(image_path)
                    display.display(display.Image(image_path))
                    print("Name: {}".format(image_path.split('/')[-1][:-4]))
                print()
                print('#########################')

            relative_path = os.path.join('/', *data_paths[0].split('/')[:-1])
            list_of_paths = list()
            label_file = self.train_file_path if data_path == self.train_dst_path else self.test_file_path
            with open(label_file) as l_file:
                for l in l_file:
                    l = l[:-1].split('\t')
                    if len(l) >= 3:
                        left_path = os.path.join(relative_path, l[0] + "_" + l[1].zfill(4) + '.jpg')
                        right = l[0] + '_' + l[2].zfill(4) if len(l) == 3 else l[2] + '_' + l[3].zfill(4)
                        right_path = os.path.join(relative_path, right + '.jpg')
                        list_of_paths.append((left_path, right_path, int(len(l) == 3)))
            train_test_list_of_paths[index] = list_of_paths
        return train_test_list_of_paths

    # -----------------------------------------------------------
    #                  PREPROCESSING UTILS
    # -----------------------------------------------------------

    def _create_pairs(self, textfile_path):
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

    def _flatten_dirs(self, src):
        """
        This functions flatten a given folder so the files won't be in the inner directory.
        :param src: the path that needs to be flattened.
        :param verbose: True if we want logging of the moving, else False.
        """
        for dir in os.listdir(src):
            for file in os.listdir(os.path.join(src, dir)):
                if self.verbose:
                    print("Moving file: " + file + "....")
                shutil.move(os.path.join(src, dir, file), os.path.join(src, file))

    def _move_dirs(self, src_path, dest_path, folders):
        """
        This functions moves the dir to another place and flatten a given folder so the files won't be in the inner directory.
        :param src_path: the source path that we will move.
        :param dest_path: the destination path that we will move to from the source_path.
        :param folders: names of the folder after the moving.
        :param verbose: True if we want logging of the moving, else False.
        """
        for folder in folders:
            if self.verbose:
                print("Moving folder: " + folder + "....")
            shutil.move(os.path.join(src_path, folder), os.path.join(dest_path, folder))
        self._flatten_dirs(src=dest_path)

    def prepare_data(self):
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
        train_names = self._create_pairs(self.train_file_path)
        test_names = self._create_pairs(self.test_file_path)
        self._move_dirs(self.src_path, self.train_dst_path, train_names)
        self._move_dirs(self.src_path, self.test_dst_path, test_names)
        self._flatten_dirs(self.src_path)
        shutil.move(self.src_path, self.unused_path)

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