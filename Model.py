import numpy as np
import time
import datetime
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Lambda, Flatten, Dense
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorboard.plugins.hparams import api as hyper
from tensorflow.random import set_seed
from numpy.random import seed
from  distanceFunc  import abs_distance
import LFWDataLoader
from skimage.transform import resize
from skimage import io
import matplotlib.pyplot as plt
from utils import get_image_name

def initialize_bias(shape, name=None, dtype=None):
    """
    initialize the weights of the biases for the network
    :param shape: the shape of the bias
    :param name: name of the weights
    :param dtype: type of the weights
    :return: the returned value by the normal distribution
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def initialize_weights(shape, name=None, dtype=None):
    """
    initialize the weights for the network
    :param shape: the shape of the weights
    :param name: name of the weights
    :param dtype: type of the weights
    :return: the returned value by the normal distribution
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_weights_dense(shape, name=None, dtype=None):
    """
    initialize weights for the dense layers
    :param shape: the shape of the weights
    :param name: name of the weights
    :param dtype: type of the weights
    :return: the returned value by the normal distribution
    """
    return np.random.normal(loc=0.0, scale=0.2, size=shape)


class SiameseModel:

    def __init__(self, input_shape=(250, 250, 1), num_layers=4, filter_size=64,
                 kernel_init=initialize_weights, kernel_init_dense=initialize_weights_dense, kernel_reg=l2(3e-4),
                 kernel_reg_dense=l2(1e-3), bias_init=initialize_bias,
                 kernel_sizes=[(10, 10), (7, 7), (4, 4), (4, 4)], units=4 * 64, optimizer='adam', lr=3e-4,
                 loss='binary_crossentropy', metrics=['accuracy', Precision(name='Precision'), Recall(name='Recall')],
                 pretrained_weights=None, model_path=None, distance=abs_distance, distance_output_shape=None,
                 activation_predict='sigmoid'):
        """
        implement the siamese model as defined in the article - Siamese Neural Network for One-Shot Image Recognition
        :param input_shape: the shape of the images
        :param num_layers: number of layers
        :param filter_size: the size of filter
        :param kernel_init: the function for initializing the weights
        :param kernel_init_dense: the function for initializing the weights for dense layer
        :param kernel_reg: the regularization function for all the layer but the Dense
        :param kernel_reg_dense: the regulatization fucntion for Dense layer
        :param bias_init: the function for initialization the biases.
        :param kernel_sizes: the size of the hidden layers
        :param units: the number of units in Dense layer
        :param optimizer: the kind of optimizer
        :param loss: the kind of loss function
        :param metrics: the metrics we'll use
        :param pretrained_weights: if given, it will contain the pretrained weights of the model. else None
        :param model_path: if given, it will save time and just load the model from model path instead of building it. else None
        :param distance: the distance function for evaluating the distance
        :param distance_output_shape: the size of the ouput of distance function
        :param activation_predict: the kind of the activation in the prediction.
        """
        if model_path is not None:
            self.model = load_model(model_path)

        # define the model
        model = Sequential()

        # define the tensors for the input images
        left_image = Input(input_shape)
        right_image = Input(input_shape)

        self.pretrained_weights = pretrained_weights
        self.model_path = model_path
        self.distance = distance
        self.distance_output_shape = distance_output_shape
        self.kernel_init = kernel_init
        self.kernel_init_dense = kernel_init_dense
        self.kernel_reg = kernel_reg
        self.kernel_reg_dense = kernel_reg_dense
        self.bias_init = bias_init
        self.kernel_sizes = kernel_sizes
        self.optimizer = optimizer
        self.activation_predict = activation_predict
        self.loss = loss
        self.metrics = metrics
        self.filter_size = filter_size
        self.units = units
        self.lr = lr

        if optimizer is None or optimizer == 'adam':
            self.optimizer = Adam(lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = SGD(lr=lr)
        elif optimizer == 'rmd':
            self.optimizer = RMSprop(lr=lr)

        self.optimizer_name = optimizer

        # First layer
        model.add(Conv2D(
            filters=filter_size,
            kernel_size=kernel_sizes[0],
            activation='relu',
            input_shape=input_shape,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name='Conv1'
        )
        )
        model.add(MaxPooling2D())

        # Second Layer
        model.add(Conv2D(
            filters=filter_size*2,
            kernel_size=kernel_sizes[1],
            activation='relu',
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            bias_initializer=bias_init,
            name='Conv2'
        )
        )
        model.add(MaxPooling2D())

        # Third Layer
        model.add(Conv2D(
            filters=filter_size * 4,
            kernel_size=kernel_sizes[2],
            activation='relu',
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            bias_initializer=bias_init,
            name='Conv3'
        )
        )
        model.add(MaxPooling2D())

        # Fourth Layer
        model.add(Conv2D(
            filters=(filter_size * 8),
            kernel_size=kernel_sizes[3],
            activation='relu',
            kernel_initializer=initialize_weights,
            kernel_regularizer=kernel_reg,
            bias_initializer=bias_init,
            name='Conv4'
        ))

         # Dense Layer
        model.add(Flatten())

        model.add(Dense(
            units=units,
            activation=activation_predict,
            kernel_initializer=kernel_init_dense,
            kernel_regularizer=kernel_reg_dense,
            bias_initializer=bias_init,
            name='Dense1'
        )
        )

        encoded_left = model(left_image)
        encoded_right = model(right_image)

        # add a customized layer to compute the absolute distances between the left and right encodings
        distance_func = Lambda(distance, distance_output_shape)([encoded_left, encoded_right])

        # add a dense layer with sigmoid as activation to generate the similarity score
        prediction = Dense(1, activation=activation_predict, bias_initializer=bias_init)(distance_func)

        siamese_model = Model(inputs=[left_image, right_image], outputs=prediction)
        siamese_model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)

        if self.pretrained_weights:
            siamese_model.load_weights(pretrained_weights)

        self.model = siamese_model

    def fit(self, train_paths_labels, val_paths_labels, table=None, _resize=[250, 250],
            norm=255.0, batch_size=128, epochs=30, verbose=False, train_data=None, validation_data=None,
            callbacks=None, steps_per_epoch=None, validation_steps=None, hparam=None, prefix='', patience=3,
            tensorboard_hist_freq=1):
        start_time = time.time()
        start_runtime = datetime.datetime.now().strftime('%m%d-%H%M%S')
        log_name = f'{prefix}_{start_runtime}_shape{_resize[0]}_batch{batch_size}_epochs{epochs}_lr{self.lr}'
        log_paths = f'fit_logs/{log_name}'

        if table is not None:
            if hparam is not None:
                table.add_row(
                    [start_runtime, log_name, _resize[0], epochs] + [hparam[param] for param in hparam])
            else:
                table.add_row(
                    [start_runtime, log_name, _resize[0], epochs, self.units, self.filter_size,
                     batch_size, self.optimizer_name, self.lr])
            print(table)

        if steps_per_epoch is None:
            steps_per_epoch = len(train_paths_labels) // batch_size
        if validation_steps is None:
            validation_steps = len(val_paths_labels) // batch_size

        print(f'Steps per epoch: {steps_per_epoch}')
        print(f'Validation Steps:{validation_steps}')

        train_data = LFWDataLoader.init_tensor_data(train_data, images_labels_path=train_paths_labels, norm=norm, _resize=_resize,batch_size=batch_size, verbose=verbose)
        validation_data = LFWDataLoader.init_tensor_data(validation_data, images_labels_path=val_paths_labels, norm=norm,_resize=_resize, batch_size=batch_size, verbose=verbose)
        if self.pretrained_weights is None:
            if callbacks is None:
                callbacks = []

            tb_callback = TensorBoard(
                log_dir=log_paths,
                histogram_freq=tensorboard_hist_freq,
            )
            early_stop = EarlyStopping(patience=patience, verbose=0, monitor='val_loss',restore_best_weights=True)
            mc = ModelCheckpoint(f'{log_name}.h5', verbose=0, save_best_only=True)

            callbacks.append(tb_callback)
            callbacks.append(early_stop)
            callbacks.append(mc)

            if hparam is not None:
                callbacks.append(hyper.KerasCallback(log_paths, hparam))
            history = self.model.fit(train_data, epochs=epochs, verbose=verbose, callbacks=callbacks,\
                           validation_data=validation_data, steps_per_epoch=steps_per_epoch,\
                           validation_steps=validation_steps, shuffle=True)
        train_time = time.time() - start_time
        print(f'############## {train_time:.2f} seconds! ##############')
        return table, train_time,history

    def evaluate(self, image_labels_path, data=None, steps=None, norm=255.0, _resize=[250, 250], verbose=False):
        data = LFWDataLoader.init_tensor_data(data, images_labels_path=image_labels_path, norm=norm, _resize=_resize)
        if steps is None:
            steps = len(image_labels_path)
        return self.model.evaluate(data, steps=steps, verbose=verbose)

    def predict(self, images_labels_path, data=None, steps=None, norm=255.0, _resize=[250, 250], images_to_print=0,verbose=False):
        data = LFWDataLoader.init_tensor_data(data, images_labels_path=images_labels_path, norm=norm, _resize=_resize, batch_size=1)
        if steps is None:
            steps = len(images_labels_path)
        preds = np.squeeze(self.model.predict(data, steps=steps, verbose=verbose))
        print(f'Shape of prediction verctor: {preds.shape}')
        split = 1
        num_of_split = 2
        plt.figure(figsize=(num_of_split * 4, images_to_print * 4))

        for i in range(images_to_print):
            left_img, right_img, y_true = images_labels_path[i]
            y_pred = preds[i]
            plt.subplot(images_to_print, num_of_split, split)
            plt.imshow(resize(io.imread(left_img, as_gray=True), (_resize[0], _resize[1])), cmap='gray', vmin=0, vmax=1)
            plt.title(f'{i}: {get_image_name(left_img)}\n true={y_true}\npred={y_pred:.2f}')
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(images_to_print, num_of_split, split)
            plt.imshow(resize(io.imread(right_img, as_gray=True), (_resize[0], _resize[1])), cmap='gray', vmin=0, vmax=1)
            plt.title(f'{i}: {get_image_name(right_img)}\n true={y_true}\npred={y_pred:.2f}')
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

            split += num_of_split
        plt.show()
        return preds

    def fit_evaluate(self, train_paths_labels, val_paths_labels, test_paths_labels, fit_table=None, eval_table=None,
                     _resize=[250, 250], norm=255.0, batch_size=128, epochs=30, verbose=False,
                     train_data=None,
                     validation_data=None,
                     callbacks=None, steps_per_epoch=None, validation_steps=None, prefix='', patience=3,
                     tensorboard_hist_freq=1, random_seed=42):
        seed(random_seed)
        set_seed(random_seed)
        _, train_time, history = self.fit(train_paths_labels=train_paths_labels, val_paths_labels=val_paths_labels,
                                 table=fit_table,
                                 _resize=_resize, norm=norm, batch_size=batch_size, epochs=epochs,
                                 verbose=verbose, train_data=train_data, validation_data=validation_data,
                                 callbacks=callbacks,
                                 steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, prefix=prefix,
                                 patience=patience,
                                 tensorboard_hist_freq=tensorboard_hist_freq)

        test_scores = self.evaluate(image_labels_path=test_paths_labels, norm=norm, _resize=_resize, verbose=verbose)
        val_scores = self.evaluate(image_labels_path=val_paths_labels, norm=norm, _resize=_resize, verbose=verbose)

        if eval_table is not None:
            eval_table.add_row(
                [prefix, f'{train_time:.2f}'] + [test_scores[0]] + [f'{s * 100:.2f}%' for s in test_scores[1:]]
                + [val_scores[0]] + [f'{s * 100:.2f}%' for s in val_scores[1:]])
            print(eval_table)

        return val_scores[1], history
