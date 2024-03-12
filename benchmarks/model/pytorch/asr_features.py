import torch
import torch.nn as nn
import math
import enum
import torch.nn.functional as F
from einops import rearrange, reduce


class WindowType(enum.Enum):
    Hamming = 0
    Hanning = 1


class FrequencyScale(enum.Enum):
    MEL = 0
    LOG10 = 1
    LINEAR = 2


def next_pow_2(n):
    return 2 ** math.ceil(math.log2(n))


def sliding_window_output_length(window, stride, input_length):
    if type(input_length) == int:
        return max(input_length - window, 0) // stride + 1
    else:
        return torch.maximum(input_length - window, torch.tensor(0)) // stride + 1


def num_sample_per_frame(sampling_freq, frame_size_ms):
    return round(sampling_freq * frame_size_ms / 1000.0)


def hertz_to_warped_scale(hz, freqscale):
    if freqscale == FrequencyScale.MEL:
        return 2595.0 * torch.log10(torch.tensor(1.0 + hz / 700.0))
    elif freqscale == FrequencyScale.LOG10:
        return torch.log10(torch.tensor(hz))
    elif freqscale == FrequencyScale.LINEAR:
        return hz
    else:
        raise ValueError("invalid freqscale")


def warped_to_hertz_scale(wrp, freqscale):
    if freqscale == FrequencyScale.MEL:
        return 700.0 * (10 ** (wrp / 2595.0) - 1)
    elif freqscale == FrequencyScale.LOG10:
        return torch.pow(10, wrp)
    elif freqscale == FrequencyScale.LINEAR:
        return wrp
    else:
        raise ValueError("invalid freqscale")


def compute_derivative(input, windowlen):
    norm = (windowlen * (windowlen + 1) * (2 * windowlen + 1)) / 3.0
    input = F.pad(input, ((windowlen, windowlen), (0, 0)), mode="edge")
    kernel = torch.arange(-windowlen, windowlen + 1)[:, None]
    return F.conv2d(input, kernel, padding="valid") / norm


def length_to_mask(lengths, max_length):
    indices = rearrange(torch.arange(max_length), "t -> 1 t").to(lengths.device)
    return indices < rearrange(lengths, "b -> b 1")  # (batch_size, max_length)


# x: NxWxC length: N
def masked_mean2d(x, x_length, return_mask=False):
    x_mask = rearrange(length_to_mask(x_length, x.shape[1]), "b t -> b t 1")
    C = x.shape[2]
    scale = 1 / (C * torch.maximum(torch.tensor(1), x_length.clone().detach()))
    scale = scale.to(x.dtype)
    mean = reduce(x * x_mask, "b t c -> b", "sum") * scale
    if return_mask:
        return mean, x_mask
    else:
        return mean


def length_masked_normalize2d(x, x_length, epsilon=1e-7, return_mask=False):
    x_mask = rearrange(length_to_mask(x_length, x.shape[1]), "b t -> b t 1")
    C = x.shape[2]
    scale = 1 / (C * torch.maximum(torch.tensor(1), x_length.clone().detach()))
    scale = scale.to(x.dtype)
    mean = reduce(x * x_mask, "b t c -> b", "sum") * scale
    x = x - rearrange(mean, "b -> b 1 1")
    norm = torch.rsqrt(reduce(x * x * x_mask, "b t c -> b", "sum") * scale + epsilon)
    x = torch.where(x_mask, x * rearrange(norm, "b -> b 1 1"), 0)
    if return_mask:
        return x, x_mask
    else:
        return x


class LengthMaskedNorm2d(nn.Module):
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.scale = torch.nn.Parameter(torch.ones((1,)))
        self.bias = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self, input, length):
        res, mask = length_masked_normalize2d(input, length, self.epsilon, True)
        res = torch.where(mask, res * self.scale + self.bias, 0)
        return res


class SlidingWindow(nn.Module):
    def __init__(self, window, stride) -> None:
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, input):
        osz = sliding_window_output_length(
            self.window, self.stride, input.shape[0]
        )
        self.idx = (
            self.stride * torch.arange(osz)[:, None]
            + torch.arange(self.window)[None, :]
        )
        return input[self.idx]


class Dither(nn.Module):
    def __init__(self, coeff=0.1) -> None:
        super().__init__()
        self.coeff = coeff

    def forward(self, input):
        return input + self.coeff * torch.normal(mean=0, std=1, size=input.shape)


class PreEmphasis(nn.Module):
    def __init__(self, coeff=1.0) -> None:
        super().__init__()
        self.coeff = coeff

    def forward(self, input):
        inputm1 = torch.roll(input, 1, -1)
        inputm1[:, 0] = input[:, 0]
        return input - self.coeff * inputm1


class Windowing(nn.Module):
    def __init__(self, window, window_type) -> None:
        super().__init__()
        if window_type == WindowType.Hamming:
            self.coeffs = torch.hamming_window(window)
        elif window_type == WindowType.Hanning:
            self.coeffs = torch.hann_window(window)
        else:
            raise ValueError("invalid windowtype")

    def forward(self, input):
        return input * self.coeffs.to(input.device)


class PowerSpectrum(nn.Module):
    def __init__(self, n_fft) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.n_fft_div2 = n_fft // 2 + 1

    def forward(self, input):
        out = torch.abs(torch.fft.fft(input, self.n_fft))
        out = out[:, : self.n_fft_div2]
        return out


class MelFilterbanks(nn.Module):
    def __init__(
        self,
        num_filters,
        filter_len,
        sampling_freq,
        low_freq=0,
        high_freq=-1,
        mel_floor=0.0,
        freq_scale=FrequencyScale.MEL,
    ) -> None:
        super().__init__()
        if high_freq <= 0:
            high_freq = sampling_freq // 2

        min_warpfreq = hertz_to_warped_scale(low_freq, freq_scale)
        max_warpfreq = hertz_to_warped_scale(high_freq, freq_scale)
        dwarp = (max_warpfreq - min_warpfreq) / (num_filters + 1)

        f = torch.arange(num_filters + 2)
        f = (
            warped_to_hertz_scale(f * dwarp + min_warpfreq, freq_scale)
            * (filter_len - 1.0)
            * 2.0
            / sampling_freq
        )

        hi_slope = torch.arange(filter_len)[:, None] - f[None, :]
        hi_slope = hi_slope / (torch.roll(f, -1) - f)
        hi_slope = hi_slope[:, :num_filters]

        lo_slope = torch.roll(f, -2)[None, :] - torch.arange(filter_len)[:, None]
        lo_slope = lo_slope / (torch.roll(f, -2) - torch.roll(f, -1))
        lo_slope = lo_slope[:, :num_filters]

        self.H = torch.maximum(torch.minimum(hi_slope, lo_slope), torch.tensor(0.0))
        self.mel_floor = torch.tensor(mel_floor)

    def forward(self, input):
        return torch.maximum(
            input @ self.H.to(input.device), self.mel_floor.to(input.device)
        )


class Derivatives(nn.Module):
    def __init__(self, window_len, dblwindow_len=0) -> None:
        super().__init__()
        self.window_len = window_len
        self.dblwindow_len = dblwindow_len

    def forward(self, input):
        res = [input]
        if self.window_len > 0:
            deltas = compute_derivative(input, self.window_len)
            res.append(deltas)
            if self.dblwindow_len > 0:
                res.append(compute_derivative(deltas, self.dblwindow_len))
        return torch.cat(res, -1)


class PostProcessing(nn.Module):
    def __init__(self, use_energy) -> None:
        super().__init__()
        self.use_energy = use_energy

    def forward(self, out, energy):
        if self.use_energy:
            out = torch.cat([energy, out], -1)
        return out


class LogMelSpectrumCalculator(nn.Module):
    def __init__(
        self,
        n_filterbank,
        sampling_freq,
        frame_size_ms=25,
        frame_stride_ms=10,
        dither_coeff=0.0,
        pre_emphasis_coeff=0.97,
        window_type=WindowType.Hamming,
        use_energy=False,
        use_energy_raw=False,
        use_power=False,
        low_freq=0,
        high_freq=-1,
        mel_floor=1.0,
        freq_scale=FrequencyScale.MEL,
        delta_window=0,
        ddelta_window=0,
        post_process=None,
    ) -> None:
        super().__init__()

        n_sample_per_frame = num_sample_per_frame(sampling_freq, frame_size_ms)
        n_sample_per_stride = num_sample_per_frame(sampling_freq, frame_stride_ms)
        self.slwin = SlidingWindow(n_sample_per_frame, n_sample_per_stride)
        n_fft = next_pow_2(n_sample_per_frame)
        self.di = Dither(dither_coeff)
        self.pe = PreEmphasis(pre_emphasis_coeff)
        self.win = Windowing(n_sample_per_frame, window_type)
        self.ps = PowerSpectrum(n_fft)
        self.mel_fnc = MelFilterbanks(
            n_filterbank,
            n_fft // 2 + 1,
            sampling_freq,
            low_freq,
            high_freq,
            mel_floor,
            freq_scale,
        )
        self.der = Derivatives(delta_window, ddelta_window)
        if not post_process:
            self.post_process = PostProcessing(use_energy)
        else:
            self.post_process = post_process

        self.use_energy = use_energy
        self.use_energy_raw = use_energy_raw
        self.dither_coeff = dither_coeff
        self.use_power = use_power

    def forward(self, input):
        out = input * 32768.0
        out = self.slwin(out)
        energy = None
        if self.use_energy and self.use_energy_raw:
            energy = torch.log(
                torch.maximum(
                    torch.sum(out * out, -1, keepdim=True),
                    torch.finfo(torch.float32).tiny,
                )
            )
        if self.dither_coeff > 0:
            out = self.di(out)
        out = self.pe(out)
        out = self.win(out)
        if self.use_energy and not self.use_energy_raw:
            energy = torch.log(
                torch.maximum(
                    torch.sum(out * out, -1, keepdim=True),
                    torch.finfo(torch.float32).tiny,
                )
            )
        out = self.ps(out)
        if self.use_power:
            out = out * out
        out = self.mel_fnc(out)
        out = torch.log(
            torch.maximum(out, torch.tensor(torch.finfo(torch.float32).tiny))
        )
        out = self.post_process(out, energy)
        out = self.der(out)
        return out
