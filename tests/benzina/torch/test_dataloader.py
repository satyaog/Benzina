import torch

import benzina.torch as bz
import benzina.torch.operations as ops
from benzina.utils import File, Track


def test_classificationdataset():
    dataset_path = "annex/mini_concat_indexed.bzna"

    dataset = bz.dataset.ClassificationDataset(dataset_path)
    dataloader = bz.dataloader.DataLoader(dataset, shape=(224, 224))

    for images, targets in dataloader:
        for i, (image, target) in enumerate(zip(images, targets)):
            assert image.size() == (3, 224, 224)
            assert image.sum().item() > 0

    with File(dataset_path) as file:
        with Track(file, "bzna_input") as input_track, \
             Track(file, "bzna_target") as target_track:
            o_dataset = bz.dataset.ClassificationDataset(tracks=(input_track,
                                                                 target_track))
            o_dataloader = bz.dataloader.DataLoader(o_dataset,
                                                    shape=(224, 224))

            for (images, targets), (o_images, o_targets) in zip(dataloader,
                                                                o_dataloader):
                assert o_images == images
                assert o_targets == targets

            indices = list(range(len(dataset)))
            fullset = torch.utils.data.Subset(dataset, indices)

            o_dataloader = bz.dataloader.DataLoader(fullset, shape=(224, 224),
                                                    path=dataset_path)

            for (images, targets), (o_images, o_targets) in zip(dataloader,
                                                                o_dataloader):
                assert o_images == images
                assert o_targets == targets


def test_classificationdataset_w_ops():
    dataset_path = "annex/mini_concat_indexed.bzna"

    dataset = bz.dataset.ClassificationDataset(dataset_path)
    dataloader = bz.dataloader.DataLoader(
        dataset, shape=(224, 224),
        warp_transform=ops.SimilarityTransform(scale=(0.08, 1.0),
                                               ratio=(3./4., 4./3.),
                                               flip_h=0.5,
                                               random_crop=True))

    for images, targets in dataloader:
        for i, (image, target) in enumerate(zip(images, targets)):
            assert image.size() == (3, 224, 224)
            assert image.sum().item() > 0
