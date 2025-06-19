# By Zhouyou 2025/03/19
from typing import Iterator
import math
from mmengine.dist import get_dist_info, sync_random_seed
from typing import Iterator, Optional, Sized, Sequence

import torch
from torch.utils.data import WeightedRandomSampler
from mmengine.dataset import DefaultSampler

from mmpretrain.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class WeightRandomSampler(DefaultSampler):
    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 weights: Sequence[float] = [], 
                 num_samples: int = None,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up
        
        self.class_weights = weights
        self.num_samples = num_samples

        self.weights = []
        #print(self.class_weights)
        for i in range(len(dataset)):
            label = dataset[i]['data_samples'].gt_label
            #print(dataset[i]['data_samples'].gt_label)
            self.weights.append(self.class_weights[int(label)])
        self.weights = torch.as_tensor(self.weights, dtype=torch.double)

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            #indices = torch.randperm(len(self.dataset), generator=g).tolist()
            sampler = WeightedRandomSampler(self.weights, self.total_size, replacement=True, generator=g)
            indices = list(sampler)
            #print(indices)
            #print(self.dataset)
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)
    

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch