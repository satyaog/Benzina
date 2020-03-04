import os

import benzina.torch as bz
from benzina.utils.file import File, Track


def test_imagenet():
    dataset_path = os.environ["DATASET_PATH"]

    with File(dataset_path) as file:
        with Track(file, "bzna_input") as input_track, \
             Track(file, "bzna_target") as target_track:
            dataset = bz.dataset.ImageNet(input_track, target_track)
            dataloader = bz.dataloader.DataLoader(dataset, dataset_path,
                                                  shape=(224, 224))

            for images, targets in dataloader:
                #
                # The targets tensor is still collated on CPU. Move it to same
                # device as images.
                #
                targets = targets.to(images.device)
                for i, (image, target) in enumerate(zip(images, targets)):
                    assert image.size() == (3, 224, 224)
                    assert image.sum().item() > 0

if __name__ == "__main__":
    test_imagenet()
