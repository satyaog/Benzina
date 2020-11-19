from benzina.torch import Dataset, ClassificationDataset
from benzina.utils import File, Track


def test_dataset():
    dataset_path = "annex/mini_concat_indexed.bzna"

    dataset = Dataset(dataset_path)

    assert dataset.filename == dataset_path

    item = dataset[2]
    sample, = item

    with Track(sample.as_file(), "bzna_input") as sub_input_track:
        # Dataset does not keep the file open for it to be serializable
        sub_input_track.open()
        assert len(sub_input_track) == 1
        assert sub_input_track.shape == (600, 535)
        assert sub_input_track.sample_location(0) == (232384, 215750)
        assert sub_input_track.video_configuration_location() == (623989, 2188)
        assert sub_input_track[0].location == sub_input_track.sample_location(0)

    with Track(item.input.as_file(), "bzna_input") as sub_input_track:
        # Dataset does not keep the file open for it to be serializable
        sub_input_track.open()
        assert len(sub_input_track) == 1
        assert sub_input_track.shape == (600, 535)
        assert sub_input_track.sample_location(0) == (232384, 215750)
        assert sub_input_track.video_configuration_location() == (623989, 2188)
        assert sub_input_track[0].location == sub_input_track.sample_location(0)

    with Track(dataset_path, "bzna_input") as track:
        assert len(dataset) == len(track)

        with Track(track.sample_as_file(2), "bzna_input") as sub_input_track:
            assert len(sub_input_track) == 1
            assert sub_input_track.shape == (600, 535)
            assert sub_input_track.sample_location(0) == (232384, 215750)
            assert sub_input_track.video_configuration_location() == (623989, 2188)
            assert sub_input_track[0].location == sub_input_track.sample_location(0)

        o_dataset = Dataset(track)

        assert o_dataset.filename == dataset.filename

        for item, o_item in zip(dataset, o_dataset):
            assert item.input.location == o_item.input.location
            assert item.input.value == o_item.input.value

        o_dataset = Dataset(track=track)

        assert o_dataset.filename == dataset.filename

        for item, o_item in zip(dataset, o_dataset):
            assert o_item.input.location == item.input.location
            assert o_item.input.value == item.input.value


def test_mp4_dataset():
    from benzina.torch.dataset import Mp4Dataset

    dataset_path = "annex/mini_concat_indexed.bzna"
    dataset = Mp4Dataset(dataset_path)

    for item, o_item in zip([item for item in dataset],
                            [item for item in dataset]):
        assert o_item.input is item.input


def test_classification_dataset():
    dataset_path = "annex/mini_concat_indexed.bzna"

    dataset = ClassificationDataset(dataset_path, input_label="bzna_input")

    item = dataset[2]
    sample, aux = item
    target, = aux

    assert sample is item.input
    assert aux is item.aux

    # Dataset does not keep the file open for it to be serializable
    sample.open()

    assert len(sample) == 1
    assert sample.shape == (600, 535)
    assert sample.sample_location(0) == (232384, 215750)
    assert sample.video_configuration_location() == (623989, 2188)
    assert sample[0].location == sample.sample_location(0)

    assert target == 1

    with File(dataset_path) as file:
        with Track(file, "bzna_input") as input_track, \
             Track(file, "bzna_target") as target_track:
            assert len(dataset) == len(input_track)

            o_dataset = ClassificationDataset((input_track, target_track),
                                              input_label="bzna_input")

            assert o_dataset.filename == dataset.filename

            for item, o_item in zip(dataset, o_dataset):
                assert o_item.input.label == item.input.label
                assert o_item.input.file.offset == item.input.file.offset
                assert all(o_item.aux == item.aux)

            o_dataset = ClassificationDataset(tracks=(input_track, target_track),
                                              input_label="bzna_input")

            assert o_dataset.filename == dataset.filename

            for item, o_item in zip(dataset, o_dataset):
                assert o_item.input.label == item.input.label
                assert o_item.input.file.offset == item.input.file.offset
                assert all(o_item.aux == item.aux)
