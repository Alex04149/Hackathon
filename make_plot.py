import numpy as np
from matplotlib import pyplot as plt

from model import MLP
from data_loader import load_data

if __name__ == "__main__":
    net = MLP(mode="rand", sizes=[469,40,40,20,20,2])
    td = load_data("training")
    t_data = load_data("test")
    e = 300
    net.SGD(training_data= td, test_data = t_data, epochs=e, mini_batch_size = 10, learning_rate = 0.75, weight_decay = 0.1)

    t = np.linspace(1,e,e)
    plt.figure().set_figwidth(12)
    plt.plot(t,[net.losses_and_evaluates_list[i][1] * 100 for i in range(e)])
    plt.title("Epochs/Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("D:\\Equipment-fault-detection-1dcnn-main\\epochs_accurate.png")
    plt.clf()
    plt.close()

    plt.figure().set_figwidth(12)
    plt.plot(t,[net.losses_and_evaluates_list[i][0]  for i in range(e)])
    plt.title("Epochs/Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("D:\\Equipment-fault-detection-1dcnn-main\\epochs_loss.png")
    plt.clf()
    plt.close()

