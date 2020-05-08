from tensorflow.keras import backend as K
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