from bitstring import ConstBitStream

from pybenzinaparse import Parser, headers

from benzina.utils import File, Track


def test_file():
    dataset_path = "annex/mini_concat_indexed.bzna"

    with File(dataset_path) as file:
        input_trak = file.trak("bzna_input")
        assert input_trak == file.trak("bzna_input\0")
        assert input_trak == file.trak(b"bzna_input")
        assert input_trak == file.trak(b"bzna_input\0")


def test_track():
    dataset_path = "annex/mini_concat_indexed.bzna"

    with Track(dataset_path, "bzna_input") as track:
        assert track.label == "bzna_input"

        with File(dataset_path) as file:
            _ = Track(file, "bzna_input")
            assert track.label == _.label
            assert track.shape() == _.shape()
            assert track.sample_location(0) == _.sample_location(0)

        with Track(dataset_path, "bzna_input\0") as _:
            assert track.label + '\0' == _.label
            assert track.shape() == _.shape()
            assert track.sample_location(0) == _.sample_location(0)
        with Track(dataset_path, b"bzna_input") as _:
            assert track.label == _.label.decode("utf-8")
            assert track.shape() == _.shape()
            assert track.sample_location(0) == _.sample_location(0)
        with Track(dataset_path, b"bzna_input\0") as _:
            assert track.label + '\0' == _.label.decode("utf-8")
            assert track.shape() == _.shape()
            assert track.sample_location(0) == _.sample_location(0)

        assert len(track) == 10
        assert track.shape() == (0, 0)
        assert track.sample_location(0) == (40, 85271)
        assert track.sample_location(1) == (85311, 147045)
        assert track.sample_location(2) == (232356, 397562)
        assert track.sample_location(3) == (629918, 100634)
        assert track.sample_location(4) == (730552, 139930)
        assert track.sample_location(5) == (870482, 125221)
        assert track.sample_location(6) == (995703, 154018)
        assert track.sample_location(7) == (1149721, 159911)
        assert track.sample_location(8) == (1309632, 180278)
        assert track.sample_location(9) == (1489910, 397143)
        assert track.sample_location(-1) == (1489910, 397143)
        assert track.sample_location(-2) == (1309632, 180278)

        assert track[0].location == track.sample_location(0)

        for i, sample in enumerate(track):
            assert sample.location == track.sample_location(i)

        assert track.video_configuration_location() is None

        with Track(track.sample_as_file(2), "bzna_input") as sub_input_track:
            assert len(sub_input_track) == 1
            assert sub_input_track.shape() == (600, 535)
            assert sub_input_track[0].location == (232384, 215750)

            assert sub_input_track.video_configuration_location() == (623989, 2188)
            assert sub_input_track[0].location == sub_input_track.sample_location(0)

            video_configuration_offset, video_configuration_size = \
                sub_input_track.video_configuration_location()

            hvcC_header = headers.BoxHeader()
            hvcC_header.start_pos = -8
            hvcC_header.type = b"hvcC"
            hvcC_header.box_size = video_configuration_size + 8

            with open(dataset_path, "rb") as disk_file:
                disk_file.seek(video_configuration_offset)
                hvcC_bs = ConstBitStream(disk_file.read(video_configuration_size) + b'\n')
            hvcC = Parser.parse_box(hvcC_bs, hvcC_header)
            hvcC.load(hvcC_bs)

            assert hvcC.padding == b''
            assert hvcC.configuration_version == 1

            assert hvcC.general_profile_space == 0
            assert hvcC.general_tier_flag == 0
            assert hvcC.general_profile_idc == 1
            assert hvcC.general_profile_compatibility_flags == 1610612736
            assert hvcC.general_constraint_indicator_flags == 158329674399744
            assert hvcC.general_level_idc == 120

            assert hvcC.min_spatial_segmentation_idc == 0
            assert hvcC.parallelism_type == 0
            assert hvcC.chroma_format == 1
            assert hvcC.bit_depth_luma_minus_8 == 0
            assert hvcC.bit_depth_chroma_minus_8 == 0

            assert hvcC.avg_frame_rate == 0
            assert hvcC.constant_frame_rate == 0
            assert hvcC.num_temporal_layers == 1
            assert hvcC.temporal_id_nested == 1
            assert hvcC.length_size_minus_one == 3
            assert hvcC.num_of_arrays == 4
            assert len(hvcC.arrays) == 4

            array = hvcC.arrays[0]
            assert array.array_completeness == 1
            assert array.nal_unit_type == 32
            assert array.num_nalus == 1
            assert len(array.nalus) == 1

            nalu = array.nalus[0]
            assert nalu.nal_unit_length == 24
            assert len(nalu.nal_unit) == 24

        with track.sample_as_file(2) as sub_file:
            sub_input_track = Track(sub_file, "bzna_input")

            assert len(sub_input_track) == 1
            assert sub_input_track.shape() == (600, 535)
            assert sub_input_track[0].location == (232384, 215750)

            assert sub_input_track.video_configuration_location() == (623989, 2188)
            assert sub_input_track[0].location == sub_input_track.sample_location(0)

        with track.sample_as_file(2) as sub_file:
            sub_target_track = Track(sub_file, "bzna_target")

            assert len(sub_target_track) == 1
            assert sub_target_track.shape() == (0, 0)
            assert sub_target_track[0].location == (623481, 8)

            assert sub_target_track.sample_bytes(0) == b"\x01\x00\x00\x00\x00\x00\x00\x00"
            assert sub_target_track[0].location == sub_target_track.sample_location(0)

            sub_thumb_track = Track(sub_file, "bzna_thumb")

            assert len(sub_thumb_track) == 1
            assert sub_thumb_track.shape() == (512, 456)
            assert sub_thumb_track[0].location == (448134, 175347)

            assert sub_thumb_track.video_configuration_location() == (627537, 2223)
            assert sub_thumb_track[0].location == sub_thumb_track.sample_location(0)