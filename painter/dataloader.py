import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator, Tuple

class ShardedDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        context_len: int,
        batch_size: int,
        shuffle_shards: bool = True,
        seed: int = 42,
        sample_without_document_pos: bool = True,
        batches_per_shard: int = 4,
    ):
        self.data_dir = data_dir
        self.context_len = context_len
        self.batch_size = batch_size
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.sample_without_document_pos = sample_without_document_pos
        self.batches_per_shard = batches_per_shard

        self.shard_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.shard_files = [f for f in self.shard_files if "metadata" not in f]

        if not self.shard_files:
            raise ValueError(f"No shard files found in {data_dir}")

        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng_seed = self.seed + self.epoch
        shard_files = self.shard_files

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            shard_files = shard_files[rank::world_size]
            rng_seed += 10 * rank

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shard_files = shard_files[worker_id::num_workers]
            rng_seed += 100 * worker_id 

        rng = np.random.default_rng(rng_seed)

        if self.shuffle_shards:
            rng.shuffle(shard_files)

        for shard_path in shard_files:
            yield from self._process_shard(shard_path, rng)

    def _process_shard(self, shard_path: str, rng: np.random.Generator):
        data = np.load(shard_path, mmap_mode="r")
        max_start = len(data) - (self.context_len + 1)
        if max_start < 0:
            return

        for _ in range(self.batches_per_shard):
            starts = rng.integers(0, max_start + 1, size=self.batch_size, dtype=np.int64)
            mask = rng.random(self.batch_size) < 0.5

            inputs = np.empty((self.batch_size, self.context_len, 2), dtype=np.int64)
            targets = np.empty((self.batch_size, self.context_len), dtype=np.int64)

            for i, start in enumerate(starts):
                end = start + self.context_len
                x = np.array(data[start:end], copy=True)
                if self.sample_without_document_pos and mask[i]:
                    x[:, 1] = 0
                inputs[i] = x
                targets[i] = data[start+1:end+1, 0]

            yield torch.from_numpy(inputs), torch.from_numpy(targets)

class InfiniteDataLoader:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)

    def set_epoch(self, epoch):
        self.dataloader.dataset.set_epoch(epoch)

def create_dataloader(data_dir: str, context_len: int, batch_size: int, num_workers: int = 4, validation: bool = False):
    split_dir = os.path.join(data_dir, "val" if validation else "train")

    dataset = ShardedDataset(
        split_dir,
        context_len=context_len,
        batch_size=batch_size,
        shuffle_shards=True,
        sample_without_document_pos=not validation,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return InfiniteDataLoader(loader)
