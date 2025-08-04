import random

class EdgeNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.cpu = round(random.uniform(2.0, 4.0), 2)  # GHz
        self.memory = round(random.uniform(4.0, 8.0), 2)  # GB
        self.delay = round(random.uniform(20.0, 100.0), 2)  # ms
        self.load = round(random.uniform(0.1, 0.9), 2)  # Load rate

    def update(self):
        # update the node's status by time
        self.cpu += round(random.uniform(-1.0, 1.0), 2)
        self.memory += round(random.uniform(-0.2, 0.2), 2)
        self.delay += round(random.uniform(-5, 5), 2)
        self.load += round(random.uniform(-0.1, 0.1), 2)

        # set the max and min limits
        self.cpu = min(max(self.cpu, 1.0), 5.0)
        self.memory = min(max(self.memory, 2.0), 16.0)
        self.delay = min(max(self.delay, 5.0), 500.0)
        self.load = min(max(self.load, 0.0), 1.0)

    def to_log(self):
        # return the node's status as a string
        return f"{self.node_id},{self.cpu:.2f},{self.memory:.2f},{self.delay:.2f},{self.load:.2f}"