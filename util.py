import numpy as np
from scipy.special import expit as sigmoid
from scipy.fft import fft
from typing import Union
from numpy.typing import NDArray
import json


def format_to_4_digits(num: int) -> str:
    return f"{num:04d}"


def get_vec(index: int) -> list[float]:
    T_index: str = format_to_4_digits(index)

    with open(f".\\data\\T{T_index}.txt", 'r') as file:
        vec: list[float] = [float(line.replace('\n', '')) for line in file.readlines()]
    
    return vec


def get_label(index: int) -> int:
    with open(".\\data\\key.txt", 'r') as file:
        labels: list[int] = [int(float(line.replace('\n', ''))) for line in file.readlines()]

    return labels[index - 1]


def values_init(filename: str="values.json") -> tuple:
    with open(filename, 'r') as file:
        data = json.load(file)

    weights = data['weights']
    biases = data['biases']
    
    w_arr = []
    b_arr = []

    for w, b in zip(weights, biases):
        w_mat = np.ndarray(shape=(len(w), len(w[0])), buffer=np.array(w))
        w_arr.append(w_mat)

        b_vec = np.ndarray(shape=(len(b), ), buffer=np.array(b))
        b_arr.append(b_vec)

    sizes = [w_arr[0].shape[1]]
    sizes = list(np.concatenate((sizes, [b.shape[0] for b in b_arr])))

    return (sizes, w_arr, b_arr)


def values_update(weights, biases, filename: str="values.json") -> None:
    w_list = []
    b_list = []

    num_layers = len(weights)

    for i in range(num_layers):
        matrix = []

        for j in range(weights[i].shape[0]):
            vec = list(weights[i][j])
            matrix.append(vec)

        w_list.append(matrix)

    for i in range(num_layers):
        vec = list(biases[i])
        b_list.append(vec)

    values_dict = {'weights': w_list, 'biases': b_list}

    with open(filename, 'w') as file:
        json.dump(values_dict, file, indent=4)


def similarity(u: NDArray, v: NDArray) -> float:
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def sigmoid_prime(z: NDArray) -> NDArray:
    return sigmoid(z) * (1 - sigmoid(z))


def SiLU(z: NDArray) -> NDArray:
    return z * sigmoid(z)


def SiLU_prime(z: NDArray) -> NDArray:
    return sigmoid(z) + z * sigmoid_prime(z)


def extract_features(vec: Union[NDArray, list], step: int) -> NDArray:
    def max_pooling(v, s) -> NDArray:
        return np.array([max(v[i:i+s]) for i in range(0, len(v), s)])

    N = len(vec)  # samples
    norm = (10 / N)
    freq_amp = norm * abs(fft(vec)[1:N//2])
    max_pooled = max_pooling(freq_amp, step)
    return max_pooled - 0.1


def argmax(nums, length) -> list[int]:
    copy_nums = [num for num in nums]
    copy_nums.sort()
    copy_nums = copy_nums[-length:-1]

    max_value = 0
    max_indecies = []
    for copy_num in copy_nums:
        for j, num in enumerate(nums):
            if copy_num == num:
                max_value = j
                break

        max_indecies.append(max_value)

    return max_indecies


if __name__ == "__main__":
    # w = [np.random.randn(10, 5) for _ in range(4)]
    # b = [np.random.randn(10, ) for _ in range(4)]

    # values_update(w, b)

    # w, b = values_init()
    s, w, b = values_init()

    print(s)
