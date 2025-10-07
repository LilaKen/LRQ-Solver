import paddle

from ppcfd.data.base_datamodule import BaseDataModule


class FakeDataset(paddle.io.Dataset):
    def __init__(self, num_points: int):
        self.num_points = num_points

    def __getitem__(self, idx: int):
        data_dict = {}

        fake_data = paddle.randn([self.num_points, 3])

        data_dict["centroids"] = fake_data
        data_dict["local_centroid"] = fake_data
        data_dict["pressure"] = fake_data[..., 0]
        data_dict["p_mean"] = paddle.zeros([1])
        data_dict["p_std"] = paddle.ones([1])
        data_dict["wss"] = fake_data
        data_dict["wss_mean"] = paddle.zeros([3])
        data_dict["wss_std"] = paddle.ones([3])
        data_dict["file_name"] = "fake_data"
        data_dict["areas"] = fake_data[..., 0]
        data_dict["normal"] = fake_data
        data_dict["reference_area"] = paddle.ones([1])
        return data_dict

    def __len__(self):
        return 1


class FakeDataModule(BaseDataModule):
    def __init__(
        self,
        num_points: int = 1e7,
    ):
        super().__init__()

        num_points = int(num_points)
        print("num_points: ", num_points)

        self.train_data = FakeDataset(num_points)
        self.test_data = FakeDataset(num_points)

    def decode(self, *args, **kwargs):
        ...