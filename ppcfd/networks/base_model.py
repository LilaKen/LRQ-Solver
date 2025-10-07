import paddle


class BaseModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        """Returns the device that the model is on."""
        return paddle.device.get_device()

    def data_dict_to_input(self, data_dict, **kwargs):
        """
        Convert data dictionary to appropriate input for the model.
        """
        return data_dict


    def loss_dict(self, data_dict, **kwargs):
        """
        Compute the loss dictionary for the model.
        """
        raise NotImplementedError

    @paddle.no_grad()
    def eval_dict(self, data_dict, **kwargs):
        """
        Compute the evaluation dictionary for the model.
        """
        raise NotImplementedError
