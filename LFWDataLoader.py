import tensorflow as tf

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
        right_img =load_image_as_tensor(paths[1], norm=norm, _resize=_resize)
        return (left_img, right_img), label

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
            tensor_data = tensor_data.map(lambda paths, label: load_images_as_tensor_with_label(paths, label, norm=norm, _resize=_resize),
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

def init_tensor_data(tensor_data, images_labels_path, norm=None, _resize=None, batch_size=1, buffer_size=None,verbose=False):
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
            labels = [p[2] for p in images_labels_path]
            tensor_data = create_tensor_dataset(img_paths=images_paths, labels=labels, norm=norm, _resize=_resize)
            tensor_data = prepare_tensor_dataset(tensor_data, batch_size=batch_size, buffer_size=buffer_size)
            if verbose:
                print(tensor_data)

        return tensor_data
