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
