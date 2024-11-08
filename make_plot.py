import numpy as np
from matplotlib import pyplot as plt

from model import MLP
from data_loader import load_data

if __name__ == "__main__":
    net = MLP(mode="rand", sizes=[469,40,40,20,20,2])
    td = load_data("training")
    test_data = load_data("test")
    epochs = 130
    net.SGD(td, test_data, epochs=epochs, mini_batch_size = 10, learning_rate = 0.75, weight_decay = 0.1)
    net.SGD(td, test_data, epochs=epochs, mini_batch_size = 10, learning_rate = 0.75)

    t = np.linspace(1,epochs,epochs)
    plt.plot(t,[net.losses_and_evaluates_list[i][1] * 100 for i in range(epochs)])
    plt.title("Epochs/Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("D:\\Equipment-fault-detection-1dcnn\\epochs_accurate.png")
    plt.clf()
    plt.close()

    plt.plot(t,[net.losses_and_evaluates_list[i][0]  for i in range(epochs)])
    plt.title("Epochs/Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("D:\\Equipment-fault-detection-1dcnn\\epochs_loss.png")
    plt.clf()
    plt.close()

