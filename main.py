from DataProcessor import *
from utils import *
from prettytable import PrettyTable
TRAIN_FILEPATH = 'pairsDevTrain.txt'
TEST_FILEPATH = 'pairsDevTest.txt'
SOURCE_PATH = 'lfw2/lfw2'
TRAIN_DST_PATH = 'data/train/'
TEST_DST_PATH = 'data/test/'
UNUSED_DST_PATH = 'data/etc/'


def main():
    prepare_data()
    train_paths = DataProcessor(TRAIN_DST_PATH, TRAIN_FILEPATH, False).load_data()
    test_paths = DataProcessor(TEST_DST_PATH, TEST_FILEPATH, False).load_data()

    """
    Splitting to train and Val
    Needs to explain why the seperation is good.
    How to split the data? 
    It is often recommended to split the training set into train set and validation set; so we use a 80-20 train-validation ratio.
    If we randomly choose a 80-20 seperation ratio we will likely get subjects included in both the training and validation sets, which will cause a problem in the training. 
    Therefore, we explore seperating the sets based on the person's name, to guarantee that the sets are independent.
     But we can't also do it by person's name since a name can appear in different pairs. 
    """
    train_same = [get_image_name(p[0]) for p in train_paths[:1100]]
    train_diff = [get_image_name(p[0]) for p in train_paths[1100:]]

    # First, let's check distribution of each letter / person
    h_same = get_histogram_of_letters(train_same, 1)
    h_diff = get_histogram_of_letters(train_diff, 1)

    t = PrettyTable([''] + list(h_same.keys()))
    t.add_row(['Same'] + list(h_same.values()))
    t.add_row(['Different'] + list(h_diff.values()))
    print(t)
    print()

    letters_selected_same = h_same['J'] + h_same['K'] + h_same['L']
    letters_selected_different = h_diff['J'] + h_diff['K'] + h_diff['L']

    t = PrettyTable(['', 'Total Size', 'Chosen Letters', 'Chosen Letters Size', 'Percentage'])
    t.add_row(['Same', sum(h_same.values()), 'J,K,L', letters_selected_same,
               f'{letters_selected_same / len(train_same) * 100:.2f}%'])
    t.add_row(['Different', sum(h_diff.values()), 'J,K,L', letters_selected_different,
               f'{letters_selected_different / len(train_diff) * 100:.2f}%'])
    t.add_row(
        ['All', sum(h_same.values()) + sum(h_diff.values()), 'A-Z', letters_selected_same + letters_selected_different,
         f'{(letters_selected_same + letters_selected_different) / (len(train_same) + len(train_diff)) * 100:.2f}%'])
    print(t)

    train_same_paths, val_same_paths = split_data_by_letters(train_paths[:1100], letters='JKL')
    train_diff_paths, val_diff_paths = split_data_by_letters(train_paths[1100:], letters='JKL')

    train_total_split = train_same_paths + train_diff_paths
    val_total_split = val_same_paths + val_diff_paths

    t = PrettyTable(['', 'Size', 'Percentage'])
    t.add_row(['Train Same', f'{len(train_same_paths)}/{len(train_paths)}',
               f'{len(train_same_paths) / len(train_paths) * 100:.2f}%'])
    t.add_row(['Train Different', f'{len(train_diff_paths)}/{len(train_paths)}',
               f'{len(train_diff_paths) / len(train_paths) * 100:.2f}%'])
    t.add_row(['Train', f'{len(train_total_split)}/{len(train_paths)}',
               f'{len(train_total_split) / len(train_paths) * 100:.2f}%'])
    t.add_row(['', '', ''])
    t.add_row(['Validation Same', f'{len(val_same_paths)}/{len(train_paths)}',
               f'{len(val_same_paths) / len(train_paths) * 100:.2f}%'])
    t.add_row(['Validation Different', f'{len(val_diff_paths)}/{len(train_paths)}',
               f'{len(val_diff_paths) / len(train_paths) * 100:.2f}%'])
    t.add_row(['Validation', f'{len(val_total_split)}/{len(train_paths)}',
               f'{len(val_total_split) / len(train_paths) * 100:.2f}%'])
    print(t)


if __name__ == "__main__":
    main()