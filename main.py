import tensorflow as tf
from tensorboard.plugins.hparams import api as hyper
from numpy.random import seed
from tensorflow.random import set_seed
SEED = 42
seed(SEED)
set_seed(SEED)
from Model import initialize_weights
from distanceFunc import abs_distance
from prettytable import MSWORD_FRIENDLY
import matplotlib.pyplot as plt
from Model import SiameseModel
from DataProcessor import *
from utils import *
from prettytable import PrettyTable

TRAIN_FILEPATH = 'pairsDevTrain.txt'
TEST_FILEPATH = 'pairsDevTest.txt'
SOURCE_PATH = 'lfw2/lfw2'
TRAIN_DST_PATH = 'data/train/'
TEST_DST_PATH = 'data/test/'
UNUSED_DST_PATH = 'data/etc/'

def plotRun(history):
        fig, axes = plt.subplots(1, 2)
        fig.set_figheight(7)
        fig.set_figwidth(14)

        # plot accuracy
        axes[0].plot(history.history['accuracy'])
        axes[0].plot(history.history['val_accuracy'])
        axes[0].set_title('model accuracy during training')
        axes[0].set_ylabel('accuracy')
        axes[0].set_xlabel('epoch')
        axes[0].legend(['training', 'validation'], loc='best')

        # plot loss
        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('loss during training')
        axes[1].set_ylabel('loss')
        axes[1].set_xlabel('epoch')
        axes[1].legend(['training', 'validation'], loc='best')

def run_hyper_params(train_total_split,val_total_split,test_paths, running_dir, hparams, prefix, table=None, eval_table=None,
                     verbose=False, kernel_initializer=initialize_weights,
                     distance=abs_distance, distance_output_shape=None,
                     loss='binary_crossentropy', metrics=['accuracy'],
                     epochs=100):

    with tf.summary.create_file_writer(running_dir).as_default():
        hyper.hparams(hparams)  # record the values used in this trial
        siamese_model = SiameseModel(filter_size=hparams[hyper.NUM_FILTERS_PARAM],
                                     units=hparams[hyper.UNITS_PARAM],
                                     input_shape=(105, 105, 1),
                                     distance=distance,
                                     distance_output_shape=distance_output_shape,
                                     loss=loss,
                                     metrics=metrics,
                                     optimizer=hparams[hyper.OPTIMIZER_PARAM],
                                     lr=hparams[hyper.LR_PARAM] * 1e-4)

        accuracy, history = siamese_model.fit_evaluate(train_paths_labels=train_total_split,
                                                       val_paths_labels=val_total_split,
                                                       test_paths_labels=test_paths,
                                                       fit_table=table,
                                                       eval_table=eval_table,
                                                       _resize=[105, 105],
                                                       batch_size = 64,
                                                       epochs=epochs,
                                                       verbose=verbose,
                                                       prefix=prefix,
                                                       callbacks=[hyper.KerasCallback(running_dir, hparams)]
                                                       )
        tf.summary.scalar(metrics, accuracy, step=1)
        return history

def main():
    dataProcessor = DataProcessor(SOURCE_PATH, TRAIN_DST_PATH, TEST_DST_PATH, UNUSED_DST_PATH, TRAIN_FILEPATH,TEST_FILEPATH)
    dataProcessor.prepare_data()
    train_paths, test_paths = dataProcessor.load_data()
    """
    Splitting to train and Val
    Needs to explain why the seperation is good.
    How to split the data? 
    It is often recommended to split the training set into train set and validation set; so we use a 80-20 train-validation ratio.
    If we randomly choose a 80-20 seperation ratio we will likely get subjects included in both the training and validation sets, which will cause a problem in the training. 
    Therefore, we explore seperating the sets based on the person's name, to guarantee that the sets are independent.
     But we can't also do it by person's name since a name can appear in different pairs. 
    """

    train_same_paths, val_same_paths = split_data_by_letters(train_paths[:1100], letters='JKL')
    train_diff_paths, val_diff_paths = split_data_by_letters(train_paths[1100:], letters='JKL')

    train_total_split = train_same_paths + train_diff_paths
    val_total_split = val_same_paths + val_diff_paths

    UNITS_PARAM = hyper.HParam('units', hyper.Discrete([512, 4096]))
    NUM_FILTERS_PARAM = hyper.HParam('filter_size', hyper.Discrete([64]))
    BATCH_SIZE_PARAM = hyper.HParam('batch_size', hyper.Discrete([32]))
    OPTIMIZER_PARAM = hyper.HParam('optimizer', hyper.Discrete(['adam']))
    LR_PARAM = hyper.HParam('lr', hyper.Discrete([1]))
    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('fit_logs/hparam_tuning').as_default():
        hyper.hparams_config(
            hparams=[
                UNITS_PARAM,
                NUM_FILTERS_PARAM,
                BATCH_SIZE_PARAM,
                OPTIMIZER_PARAM,
                LR_PARAM
            ],
            metrics=[
                hyper.Metric(METRIC_ACCURACY, display_name='Accuracy'),
            ],
        )
    session_num = 3
    model_table = PrettyTable(
        ['Start_Runtime', 'Name', 'Resize', 'Epochs', 'Units', 'Filters', 'Batch Size', 'Optimizer', 'LR'])
    model_table.set_style(MSWORD_FRIENDLY)
    eval_table = PrettyTable([
        'Name',
        'Train_Time', 'Test_Loss', 'Test_Accuracy',
        'Val_Loss', 'Val_Accuracy'
    ])
    eval_table.set_style(MSWORD_FRIENDLY)

    for num_units in UNITS_PARAM.domain.values:
        for filters in NUM_FILTERS_PARAM.domain.values:
            for optimizer in OPTIMIZER_PARAM.domain.values:
                for lr in LR_PARAM.domain.values:
                    for batch_size in BATCH_SIZE_PARAM.domain.values:
                        #           for dropout_rate in HP_DROPOUT.domain.values:
                        hparams = {
                            UNITS_PARAM: num_units,
                            NUM_FILTERS_PARAM: filters,
                            BATCH_SIZE_PARAM: batch_size,
                            OPTIMIZER_PARAM: optimizer,
                            LR_PARAM: lr
                        }
                        run_title = f'Run_{session_num}'
                        print(f'--- Starting Running: {run_title} --- ')
                        print({h.name: hparams[h] for h in hparams})
                        history = run_hyper_params(
                            train_total_split, val_total_split, test_paths,
                            running_dir=f'fit_logs/hparam_tuning/{run_title}',
                            hparams=hparams,
                            prefix=f'hparam_{session_num}_all',
                            table=model_table,
                            eval_table=eval_table,
                            epochs=110,
                            verbose=True
                        )
                        session_num = session_num + 1
                        plotRun(history)

if __name__ == "__main__":
    main()