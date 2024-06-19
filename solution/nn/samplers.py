import numpy as np
from torch.utils import data


class HardMiningSampler(data.Sampler):
    def __init__(self, dataset, history_per_term=10):
        super().__init__(dataset)
        self.dataset = dataset
        self.dataset_len = int(len(dataset) / dataset.num_patches)
        self.history_per_term = history_per_term
        self._loss_history = np.zeros(
            [self.dataset_len, history_per_term],
            dtype=np.float64
        )
        self._loss_counts = np.zeros([self.dataset_len],
                                     dtype=np.int64)

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

    def update_with_local_losses(self, indices, losses):
        for idx, loss in zip(indices, losses):
            loss = float(loss.detach().cpu().numpy())
            if self._loss_counts[idx] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[idx, :-1] = self._loss_history[idx, 1:]
                self._loss_history[idx, -1] = loss
            else:
                self._loss_history[idx, self._loss_counts[idx] % self._loss_history.shape[1]] = loss
                self._loss_counts[idx] += 1

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.dataset_len], dtype=np.float64) / self.dataset_len
        weights = np.mean(self._loss_history ** 2, axis=-1)
        weights /= np.sum(weights)
        return weights

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        while True:
            yield np.random.choice(self.dataset_len, p=self.weights()) * self.dataset.num_patches

    def __next__(self):
        return np.random.choice(self.dataset_len, p=self.weights()) * self.dataset.num_patches

    # returns Index, Batch
    def __call__(self):
        return next(self)