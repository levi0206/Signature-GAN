import torch

# The common variables of different GANs
class Base:
    def __init__(self, config, G, test_metrics_train, test_metrics_test):

        self.batch_size = config['batch_size']
        self.G_optimizer = config['G_optimizer']
        self.epoch = config['epoch']

        self.G = G
        self.losses_history = []
        self.test_metrics_train = test_metrics_train
        self.test_metrics_test = test_metrics_test