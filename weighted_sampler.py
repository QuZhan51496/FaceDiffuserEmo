from torch.utils.data import Sampler
import numpy as np
from collections import Counter
from typing import Iterator, Optional, Sized
import math

class WeightedSamplerBySpkID(Sampler):
    def __init__(
        self,
        dataset,
        sample_len=None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dataset (Dataset): Dataset to sample from.
            sample_name_to_weight (dict): A dictionary mapping from the sample
                names to their corresponding sampling weights.
        """
        self.dataset = dataset
        self.sample_len = sample_len if sample_len is not None else len(dataset)
        self.seed = seed
        self.epoch = 0
        self.spk_ids = None

        # Compute weights for each sample in the dataset
        self.weights = self._compute_weights()

    def _get_spk_ids(self):
        spk_ids = [x[4] for x in self.dataset.data]
        return spk_ids

    def _compute_weights(self):
        spk_ids = self._get_spk_ids()
        id_counts = Counter(spk_ids)
        id_weights = {
            spk_id: 1.0 / count * (1 + math.log(count))
            for spk_id, count in id_counts.items()
        }
        sample_weights = [id_weights[spk_id] for spk_id in spk_ids]
        self.spk_ids = spk_ids
        return sample_weights

    def __iter__(self):
        indices = np.random.choice(
            len(self.dataset),
            size=self.sample_len,
            p=self._normalize_weights(self.weights)
        )
        indices_sampled = indices.tolist()
        return iter(indices.tolist())

    def __len__(self):
        return self.sample_len

    def _normalize_weights(self, weights):
        # Converts the weights to probabilities that sum up to 1
        total_weight = sum(weights)
        return [w / total_weight for w in weights]

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
