# source code comes from https://github.com/rlcode/per
import random
import numpy as np
from SumTree import SumTree

class memory:  # stored state_next = torch.zeros((self.batch_size, len(data[0][3]))).to(self.device)as ( s, a, r, s_ ) in SumTree

    def __init__(self, capacity):
        self.e = 0.01
        self.a = 0.5
        self.beta = 1.0
        self.beta_increment_per_sampling = 0.001

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a
    def __len__(self):
        return self.tree.n_entries
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)