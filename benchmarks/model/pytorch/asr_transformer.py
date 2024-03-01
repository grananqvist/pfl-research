import numpy as np
from torchsummary import summary

from pfl.internal.ops.pytorch_ops import get_default_device
from pfl.metrics import Summed, Weighted


def make_pytorch_dummy_model(size: int = 2):
    import torch  # type: ignore
    torch.manual_seed(42)
    from pfl.model.pytorch import PyTorchModel

    class TestModel(torch.nn.Module):

        def __init__(self, size: int):
            super().__init__()
            self.input_size = 900
            self.output_size = 100
            self.linear_relu_stack = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, size * 1000),
                torch.nn.ReLU(),
                torch.nn.Linear(size * 1000, self.output_size),
            )

        def forward(self, input, target, input_length, target_length, audio,
                    audio_file, user_id, transcript):  # pylint: disable=arguments-differ
            # a1 = torch.nn.functional.sigmoid(
            #     torch.matmul(x, self.w1) + self.b1)
            # a2 = torch.nn.functional.sigmoid(
            #     torch.matmul(a1, self.w2) + self.b2)
            # print('x:', x)
            batch_size = input.shape[0]
            x = torch.rand(
                (batch_size, self.input_size)).to(get_default_device())
            return self.linear_relu_stack(x)
            # return torch.norm(torch.flatten(self.w1))

        def loss(self,
                 input,
                 target,
                 input_length,
                 target_length,
                 audio,
                 audio_file,
                 user_id,
                 transcript,
                 is_eval=False):
            if is_eval:
                self.eval()
            else:
                self.train()
            mseloss = torch.nn.MSELoss()  #reduction='sum')
            output = self(input, target, input_length, target_length, audio,
                          audio_file, user_id, transcript)
            # print('output:', output)
            desired_output = torch.zeros(output.size()).to(
                get_default_device())
            # print('desired_output:', desired_output)
            return mseloss(output, desired_output)

        @torch.no_grad()
        def metrics(self, input, target, input_length, target_length, audio,
                    audio_file, user_id, transcript):
            loss_value = self.loss(input,
                                   target,
                                   input_length,
                                   target_length,
                                   audio,
                                   audio_file,
                                   user_id,
                                   transcript,
                                   is_eval=True)
            num_samples = 1
            # print('input.shape:', input.shape, np.prod(input.shape))
            output = {
                'loss': Weighted(loss_value, num_samples),
                'num_samples': Summed(num_samples),
            }
            # print('calculated metrics:', output)
            return output

    pytorch_model = TestModel(size).to(get_default_device())

    print('pytorch_model:', pytorch_model)
    pytorch_total_params = sum(p.numel() for p in pytorch_model.parameters())
    print('Total params:', pytorch_total_params)

    from prettytable import PrettyTable

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    count_parameters(pytorch_model)

    return pytorch_model
