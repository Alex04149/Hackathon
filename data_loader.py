import numpy as np
from numpy.typing import NDArray

from util import get_vec, get_label, extract_features
from errors import OptionError


class LoadDataset:
    def __init__(self) -> None:
        raise NotImplementedError()


def load_data(option: str) -> list:
    options = ('training', 'test', 'validation')

    training_data = []
    test_data = []
    validation_data = []
    step = 100

    def vectorize(j, size=2) -> NDArray:
        vec = np.zeros(size, )
        vec[j] = 1
        return vec

    def load_pair(index) -> tuple[NDArray, NDArray]:
        vec, label = extract_features(get_vec(index), step), vectorize(get_label(index))
        pair = (vec, label)
        return pair

    match option:
        case 'training':
            for i in range(1, 101):
                training_data.append(load_pair(i))

            return training_data
        case 'test':
            for i in range(1, 1159):
                test_data.append(load_pair(i))

            return test_data
        case 'validation':
            for i in range(1158, 1058, -1):
                validation_data.append(load_pair(i))

            return validation_data
        case _:
            raise OptionError(f"Incorrect option. Should be {options}")


if __name__ == "__main__":
    vecs = load_data('training')

    print([vecs[i][1] for i in range(100)])

    # ef_vecs = [extract_features(vecs[i][0], step) for i in range(len(vecs))]

    # print(vecs[0][0])
