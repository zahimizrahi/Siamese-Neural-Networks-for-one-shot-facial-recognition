import shutil
import os
import IPython.display as display
import numpy as np
import tensorflow as tf
import pathlib


class DataProcessor():
    def __init__(self, data_path, label_file, verbose=True):
        """
        Constructor of DataProcessor.
        :param data_path:
        :param label_file:
        :param print_imgs:
        """
        self.data_path = data_path
        self.label_file = label_file
        self.verbose = verbose

    def load_data(self, count_print_imgs=3):
        full_path = pathlib.Path(self.data_path)
        data_paths = list(full_path.glob('*'))
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

        relative_path = os.path.join('/', os.path.commonprefix(data_paths))
        list_of_paths = list()
        with open(self.label_file) as label_file:
            for l in label_file:
                l = l[:-1].split('\t')
                if len(l) >= 3:
                    left_path = os.path.join(relative_path, l[0] + "_" + l[1].zfill(4) + '.jpg')
                    right = l[0] + '_' + l[2].zfill(4) if len(l) == 3 else l[2] + '_' + l[3].zfill(4)
                    right_path = os.path.join(relative_path, right + '.jpg')
                    list_of_paths.append((left_path, right_path))
        return list_of_paths