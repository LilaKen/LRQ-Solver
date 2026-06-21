from torch.utils.data import DataLoader


class BaseDataModule:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.val_data = None

    def dataloader(self, dataset, **kwargs):
        kwargs.pop("return_list", None)
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(dataset, collate_fn=collate_fn, **kwargs)

    def train_dataloader(self, **kwargs) -> DataLoader:
        return self.dataloader(self.train_data, **kwargs)

    def val_dataloader(self, **kwargs) -> DataLoader:
        return self.dataloader(self.val_data, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return self.dataloader(self.test_data, **kwargs)
