def create_loader(data, batch_size, validate=False):
    """Create a dataloader."""

    if validate:
        batch_size = data.shape[0]
    else:
        batch_size = individual["batch_size_train"]

    dataset = data_utils.TensorDataset(
        *[
            torch.Tensor(data[:, :2048]),
            torch.Tensor(data[:, 2048:2058]),
            torch.Tensor(data[:, 2058:]),
        ]
    )
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader