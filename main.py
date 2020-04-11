from DataProcessor import *
from utils import *
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


if __name__ == "__main__":
    main()