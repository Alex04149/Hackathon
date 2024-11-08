from model import MLP
from data_loader import load_data


net = MLP(mode='rand')

net.SGD(load_data('training'), epochs=400, mini_batch_size=10, learning_rate=2)
