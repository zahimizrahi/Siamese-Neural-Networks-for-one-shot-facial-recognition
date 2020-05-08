from string import ascii_lowercase, ascii_uppercase
import itertools

def get_image_name(path):
    return path.split('/')[-1].split('.')[0]

def get_histogram_of_letters(people_names, num_letters=1):
    combination = list(ascii_uppercase)
    lower = ascii_lowercase
    if num_letters > 1:
        for i in range(1, num_letters):
            combination = [x + y for x, y in itertools.product(combination, lower)]
    hist = dict()
    for elem in combination:
        hist[elem] = sum(p.startswith(elem) for p in people_names)
    return hist

def subset_sum(numbers, desired_length, target, delta, partial=[], result=[]):
    s = sum(partial)

    # check if the partial sum is equals to target
    if s in range(target - delta, target + delta) and len(partial) == desired_length:
        result.append(partial)
    if s >= target + delta:
        return  # if we reach the number why bother to continue

    for i in range(len(numbers)):
        n = numbers[i]
        remaining = numbers[i + 1:]
        subset_sum(remaining, desired_length, target, delta, partial + [n])
    return result

def split_data_by_letters(labels, letters):
    """
    :param labels:
    :param letters:
    :return:
    """
    train = []
    val = []
    for paths in labels:
        left = get_image_name(paths[0])
        train_append = True
        for letter in letters:
            if left.startswith(letter):
                train_append = False
                break;
        if train_append:
            train.append(paths)
        else:
            val.append(paths)
    return train, val

# CHECK IF NEED IT
def split_train_test_tensor(tensor_data, data_len, test_size=0.2):
    """
    split the tensor to train and test tensor
    :param tensor_data:
    :param data_len:
    :param test_size:
    :return:
    """
    test_data_len = int(test_size * data_len)
    test_data = tensor_data.take(test_data_len)
    train_data = tensor_data.skip(test_data_len)
    return train_data, test_data