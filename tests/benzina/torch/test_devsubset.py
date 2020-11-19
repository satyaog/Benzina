import os

import torch

import benzina.torch as bz
from benzina.utils import Track
from benzina.utils.input_access import get_indices_by_names


def test_devsubset_pytorch_loading():
    dataset_path = os.environ["DATASET_PATH"]

    with open("annex/devsubset_files", 'r') as devsubset_list:
        subset_filenames = []
        subset_targets = []
        # skip the end line following the filename
        for line in devsubset_list:
            if not line:
                continue
            subset_filenames.append(line.rstrip()[5:])
            subset_targets.append(int(line[:4]))

    dataset = bz.dataset.ImageNet(dataset_path)

    with Track(dataset_path, "bzna_fname") as filename_track:
        subset_indices = get_indices_by_names(filename_track, subset_filenames)
    subset = torch.utils.data.Subset(dataset, subset_indices)

    subset_loader = bz.DataLoader(subset, shape=(256, 256), path=dataset_path,
                                  batch_size=100, seed=0)

    for batch_i, (images, targets) in enumerate(subset_loader):
        for i, (image, target) in enumerate(zip(images, targets)):
            assert image.size() == (3, 256, 256)
            assert image.sum().item() > 0
            assert target.item() == subset_targets[batch_i * 100 + i]
