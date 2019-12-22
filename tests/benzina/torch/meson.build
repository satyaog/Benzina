pytest = find_program('pytest')

test_dataloader = files('test_dataloader.py')
test_dataset = files('test_dataset.py')

test('torch.dataloader', pytest, suite: 'torch', args: [test_dataloader])
test('torch.dataset', pytest, suite: 'torch', args: [test_dataset])
