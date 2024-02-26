import numpy as np
from pfl.metrics import Weighted, Summed
from pfl.internal.ops.pytorch_ops import get_default_device


def make_pytorch_dummy_model():
    import torch  # type: ignore
    torch.manual_seed(42)
    from pfl.model.pytorch import PyTorchModel

    class TestModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.parameter.Parameter(
                torch.tensor(np.random.normal(scale=0.1, size=(1, 100)),
                             dtype=torch.float32,
                             device='cpu'))

        def forward(self, input, target, input_length, target_length, audio, audio_file, user_id, transcript):  # pylint: disable=arguments-differ
            # a1 = torch.nn.functional.sigmoid(
            #     torch.matmul(x, self.w1) + self.b1)
            # a2 = torch.nn.functional.sigmoid(
            #     torch.matmul(a1, self.w2) + self.b2)
            # print('x:', x)
            # batch_size = input.shape[0]
            return torch.norm(torch.flatten(self.w1))

        def loss(self, input, target, input_length, target_length, audio, audio_file, user_id, transcript, is_eval=False):
            if is_eval:
                self.eval()
            else:
                self.train()
            l1loss = torch.nn.BCELoss()#reduction='sum')
            output = self(input, target, input_length, target_length, audio, audio_file, user_id, transcript)
            # print('output:', output)
            desired_output = torch.as_tensor(0.0).to(get_default_device())
            # print('desired_output:', desired_output)
            return l1loss(output, desired_output)

        @torch.no_grad()
        def metrics(self, input, target, input_length, target_length, audio, audio_file, user_id, transcript):
            loss_value = self.loss(input, target, input_length, target_length, audio, audio_file, user_id, transcript, is_eval=True)
            num_samples = 1
            # print('input.shape:', input.shape, np.prod(input.shape))
            output = {
                'loss': Weighted(loss_value, num_samples),
                'num_samples': Summed(num_samples),
            }
            # print('calculated metrics:', output)
            return output


    pytorch_model = TestModel().to(get_default_device())

    return pytorch_model