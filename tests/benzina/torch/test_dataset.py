from benzina.torch import Dataset, ImageNet
from benzina.utils import File, Track


def test_dataset():
    dataset_path = "annex/mini_concat_indexed.bzna"

    with Track(dataset_path, "bzna_input") as track:
        dataset = Dataset(track)

        assert len(dataset) == len(track)

        item = dataset[2]
        sample, = item

        with Track(sample.as_file(), "bzna_input") as sub_input_track:
            assert len(sub_input_track) == 1
            assert sub_input_track.shape == (600, 535)
            assert sub_input_track.sample_location(0) == (232384, 215750)
            assert sub_input_track.video_configuration_location() == (623989, 2188)
            assert sub_input_track[0].location == sub_input_track.sample_location(0)

        with Track(item.input.as_file(), "bzna_input") as sub_input_track:
            assert len(sub_input_track) == 1
            assert sub_input_track.shape == (600, 535)
            assert sub_input_track.sample_location(0) == (232384, 215750)
            assert sub_input_track.video_configuration_location() == (623989, 2188)
            assert sub_input_track[0].location == sub_input_track.sample_location(0)

        with Track(track.sample_as_file(2), "bzna_input") as sub_input_track:
            assert len(sub_input_track) == 1
            assert sub_input_track.shape == (600, 535)
            assert sub_input_track.sample_location(0) == (232384, 215750)
            assert sub_input_track.video_configuration_location() == (623989, 2188)
            assert sub_input_track[0].location == sub_input_track.sample_location(0)


def test_imagenet():
    dataset_path = "annex/mini_concat_indexed.bzna"

    with File(dataset_path) as file:
        with Track(file, "bzna_input") as input_track, \
             Track(file, "bzna_target") as target_track:
            dataset = ImageNet(input_track, target_track)

            assert len(dataset) == len(input_track)

            item = dataset[2]
            sample, target = item

            with Track(sample.as_file(), "bzna_input") as sample_input_track:
                assert len(sample_input_track) == 1
                assert sample_input_track.shape == (600, 535)
                assert sample_input_track.sample_location(0) == (232384, 215750)
                assert sample_input_track.video_configuration_location() == (623989, 2188)
                assert sample_input_track[0].location == sample_input_track.sample_location(0)

            with Track(item.input.as_file(), "bzna_input") as sample_input_track:
                assert len(sample_input_track) == 1
                assert sample_input_track.shape == (600, 535)
                assert sample_input_track.sample_location(0) == (232384, 215750)
                assert sample_input_track.video_configuration_location() == (623989, 2188)
                assert sample_input_track[0].location == sample_input_track.sample_location(0)

            assert target == (1,)
            assert item.target == (1,)
