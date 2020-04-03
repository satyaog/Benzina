
def get_indices_by_names(track, filenames):
    """
    Retreive the indices of inputs by file names

    Args:
        track (benzina.torch.utils.Track): track from which to fetch the indices
        filenames (sequence): a sequence of file names
    """
    filenames_indices = {}
    filenames_lookup = set(filenames)

    for i in range(len(track)):
        # strip the null char following the filename
        name = track.sample_bytes(i).decode("utf-8").rstrip()
        if name in filenames_lookup:
            filenames_indices[name] = i

    return [filenames_indices.get(filename, None) for filename in filenames]
