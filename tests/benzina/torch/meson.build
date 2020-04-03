pytest = find_program('pytest')

test_dataloader = files('test_dataloader.py')
test_dataset = files('test_dataset.py')
test_devsubset = files('test_devsubset.py')
test_operations = files('test_operations.py')

test('torch.dataloader', pytest, suite: 'torch', args: [test_dataloader], workdir: meson.source_root())
test('torch.dataset', pytest, suite: 'torch', args: [test_dataset], workdir: meson.source_root())
test('Parse devsubset', pytest, suite: 'torch', args: [test_devsubset], timeout: 120, workdir: meson.source_root())
test('torch.operations', pytest, suite: 'torch', args: [test_operations], workdir: meson.source_root())
