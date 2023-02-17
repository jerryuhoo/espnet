# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""DDSP Modules.

This code is modified from https://github.com/zhangyongmao/VISinger2/blob/master/egs/visinger2/models.py.

"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm, remove_weight_norm

from espnet2.gan_tts.hifigan.residual_block import ResidualBlock
from espnet2.layers.stft import TorchSTFT
from espnet2.gan_svs.ddsp.modules import ResBlock1, ConvReluNorm, scale_function, remove_above_nyquist, upsample, init_weights, ResBlock3, LRELU_SLOPE

class Generator_Harm(torch.nn.Module):
    def __init__(self, hps):
        super(Generator_Harm, self).__init__()
        self.hps = hps
        
        self.prenet = torch.nn.Conv1d(hps.model.hidden_channels, hps.model.hidden_channels, 3, padding=1)
                
        self.net = ConvReluNorm(hps.model.hidden_channels,
                                    hps.model.hidden_channels,
                                    hps.model.hidden_channels,
                                    hps.model.kernel_size,
                                    8,
                                    hps.model.p_dropout)

        #self.rnn = nn.LSTM(input_size=hps.model.hidden_channels, 
        #    hidden_size=hps.model.hidden_channels,
        #    num_layers=1, 
        #    bias=True, 
        #    batch_first=True, 
        #    dropout=0.5,
        #    bidirectional=True)
        self.postnet = torch.nn.Conv1d(hps.model.hidden_channels, hps.model.n_harmonic+1, 3, padding=1)
    
    def forward(self, f0, harm, mask):
        pitch = f0.transpose(1, 2)
        harm = self.prenet(harm)
        
        harm = self.net(harm) * mask
        #harm = harm.transpose(1, 2)
        #harm, (hs, hc) = self.rnn(harm)
        #harm = harm.transpose(1, 2)

        harm = self.postnet(harm)
        harm = harm.transpose(1, 2)
        param = harm
        
        param = scale_function(param)
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.hps.data.sample_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.hps.data.hop_size)
        pitch = upsample(pitch, self.hps.data.hop_size)
        
        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.hps.data.sample_rate, 1)
        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
        signal_harmonics = (torch.sin(omegas) * amplitudes)
        signal_harmonics = signal_harmonics.transpose(1, 2)
        return signal_harmonics
    
class Generator(torch.nn.Module):
    def __init__(self, hps, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, n_speakers=0, spk_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = torch.nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.upsample_rates = upsample_rates
        self.n_speakers = n_speakers

        resblock = ResBlock1
        
        self.downs = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            i = len(upsample_rates) - 1 - i
            u = upsample_rates[i]
            k = upsample_kernel_sizes[i]
            #print("down: ",upsample_initial_channel//(2**(i+1))," -> ", upsample_initial_channel//(2**i))
            self.downs.append(weight_norm(
                torch.nn.Conv1d(hps.model.n_harmonic + 2, hps.model.n_harmonic + 2,
                       k, u, padding=k//2)))
        
        self.resblocks_downs = torch.nn.ModuleList()
        for i in range(len(self.downs)):
            j = len(upsample_rates) - 1 - i
            self.resblocks_downs.append(ResBlock3(hps.model.n_harmonic + 2, 3, (1, 3)))

        
        self.concat_pre = torch.nn.Conv1d(upsample_initial_channel + hps.model.n_harmonic + 2, upsample_initial_channel, 3, 1, padding=1)
        self.concat_conv = torch.nn.ModuleList()
        for i in range(len(upsample_rates)):
          ch = upsample_initial_channel//(2**(i+1))
          self.concat_conv.append(torch.nn.Conv1d(ch + hps.model.n_harmonic + 2, ch, 3, 1, padding=1, bias=False))


        self.ups = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                torch.nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = torch.nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = torch.nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if self.n_speakers != 0:
            self.cond = torch.nn.Conv1d(spk_channels, upsample_initial_channel, 1)
            
    def forward(self, x, ddsp, g=None):

        x = self.conv_pre(x)

        if g is not None:
          x = x + self.cond(g)

        se = ddsp
        res_features = [se]
        for i in range(self.num_upsamples):
            in_size = se.size(2)
            se = self.downs[i](se)
            se = self.resblocks_downs[i](se)
            up_rate = self.upsample_rates[self.num_upsamples - 1 - i]
            se = se[:, :, : in_size // up_rate]
            res_features.append(se)

        x = torch.cat([x, se], 1)
        x = self.concat_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            in_size = x.size(2)
            x = self.ups[i](x)
            x = x[:, :, : in_size * self.upsample_rates[i]]

            x = torch.cat([x, res_features[self.num_upsamples - 1 - i]], 1)
            x = self.concat_conv[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
            
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

class Generator_Noise(torch.nn.Module):
    def __init__(self, hps):
        super(Generator_Noise, self).__init__()
        self.hps = hps
        self.win_size = hps.data.win_size
        self.hop_size = hps.data.hop_size
        self.fft_size = hps.data.n_fft
        self.istft_pre = torch.nn.Conv1d(hps.model.hidden_channels, hps.model.hidden_channels, 3, padding=1)
        
        self.net = ConvReluNorm(hps.model.hidden_channels,
                                    hps.model.hidden_channels,
                                    hps.model.hidden_channels,
                                    hps.model.kernel_size,
                                    8,
                                    hps.model.p_dropout)

        self.istft_amplitude = torch.nn.Conv1d(hps.model.hidden_channels, self.fft_size//2+1, 1, 1)
        self.window = torch.hann_window(self.win_size)

    def forward(self, x, mask):
        istft_x = x
        istft_x = self.istft_pre(istft_x)

        istft_x = self.net(istft_x) * mask
        
        amp = self.istft_amplitude(istft_x).unsqueeze(-1)
        phase = (torch.rand(amp.shape) * 2 * 3.14 - 3.14).to(amp)

        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        spec = torch.cat([real, imag], 3)
        istft_x = torch.istft(spec, self.fft_size, self.hop_size, self.win_size, self.window.to(amp), True, length=x.shape[2] * self.hop_size, return_complex=False)
    
        return istft_x.unsqueeze(1)


class DDSPGenerator(torch.nn.Module):
    """DDSP generator module."""

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        channels: int = 512,
        global_channels: int = -1,
        kernel_size: int = 7,
        upsample_scales: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        use_additional_convs: bool = True,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.1},
        use_weight_norm: bool = True,
    ):
        """Initialize DDSPGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            global_channels (int): Number of global conditioning channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (List[int]): List of upsampling scales.
            upsample_kernel_sizes (List[int]): List of kernel sizes for upsample layers.
            resblock_kernel_sizes (List[int]): List of kernel sizes for residual blocks.
            resblock_dilations (List[List[int]]): List of list of dilations for residual
                blocks.
            use_additional_convs (bool): Whether to use additional conv layers in
                residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.upsample_factor = int(np.prod(upsample_scales) * out_channels)
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.output_conv = torch.nn.Sequential(
            # NOTE(kan-bayashi): follow official implementation but why
            #   using different slope parameter here? (0.1 vs. 0.01)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )
        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, channels, 1)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(
        self, c: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        c = self.input_conv(c)
        if g is not None:
            c = c + self.global_conv(g)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def inference(
        self, c: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform inference.

        Args:
            c (torch.Tensor): Input tensor (T, in_channels).
            g (Optional[Tensor]): Global conditioning tensor (global_channels, 1).

        Returns:
            Tensor: Output tensor (T ** upsample_factor, out_channels).

        """
        if g is not None:
            g = g.unsqueeze(0)
        c = self.forward(c.transpose(1, 0).unsqueeze(0), g=g)
        return c.squeeze(0).transpose(1, 0)


class DDSPPeriodDiscriminator(torch.nn.Module):
    """DDSP period discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        period: int = 3,
        kernel_sizes: List[int] = [5, 3],
        channels: int = 32,
        downsample_scales: List[int] = [3, 3, 3, 3, 1],
        max_downsample_channels: int = 1024,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.1},
        use_weight_norm: bool = True,
        use_spectral_norm: bool = False,
    ):
        """Initialize DDSPPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv
                layer.
            channels (int): Number of initial channels.
            downsample_scales (List[int]): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            use_additional_convs (bool): Whether to use additional conv layers in
                residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = torch.nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            list: List of each layer's tensors.

        """
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs += [x]
        x = self.output_conv(x)
        x = torch.flatten(x, 1, -1)
        outs += [x]

        return outs

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class DDSPMultiPeriodDiscriminator(torch.nn.Module):
    """DDSP multi-period discriminator module."""

    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize DDSPMultiPeriodDiscriminator module.

        Args:
            periods (List[int]): List of periods.
            discriminator_params (Dict[str, Any]): Parameters for hifi-gan period
                discriminator module. The period parameter will be overwritten.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [DDSPPeriodDiscriminator(**params)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each
                layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]

        return outs



class DDSPScaleDiscriminator(torch.nn.Module):
    """HiFi-GAN scale discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: List[int] = [15, 41, 5, 3],
        channels: int = 128,
        max_downsample_channels: int = 1024,
        max_groups: int = 16,
        bias: int = True,
        downsample_scales: List[int] = [2, 2, 4, 4, 1],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.1},
        use_weight_norm: bool = True,
        use_spectral_norm: bool = False,
    ):
        """Initilize DDSP scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (List[int]): List of four kernel sizes. The first will be used
                for the first conv layer, and the second is for downsampling part, and
                the remaining two are for the last two output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling
                layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (List[int]): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm. If set to true, it
                will be applied to all of the conv layers.

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels,
                    channels,
                    # NOTE(kan-bayashi): Use always the same kernel size
                    kernel_sizes[0],
                    bias=bias,
                    padding=(kernel_sizes[0] - 1) // 2,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        out_chs = channels
        # NOTE(kan-bayashi): Remove hard coding?
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[1],
                        stride=downsample_scale,
                        padding=(kernel_sizes[1] - 1) // 2,
                        groups=groups,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Remove hard coding?
            out_chs = min(in_chs * 2, max_downsample_channels)
            # NOTE(kan-bayashi): Remove hard coding?
            groups = min(groups * 4, max_groups)

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_sizes[2],
                    stride=1,
                    padding=(kernel_sizes[2] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=(kernel_sizes[3] - 1) // 2,
                bias=bias,
            ),
        ]

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[Tensor]: List of output tensors of each layer.

        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
                logging.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class DDSPMultiScaleDiscriminator(torch.nn.Module):
    """HiFi-GAN multi-scale discriminator module."""

    def __init__(
        self,
        scales: int = 3,
        downsample_pooling: str = "AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm: bool = False,
    ):
        """Initilize DDSP multi-scale discriminator module.

        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            downsample_pooling_params (Dict[str, Any]): Parameters for the above pooling
                module.
            discriminator_params (Dict[str, Any]): Parameters for hifi-gan scale
                discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the
                official implementaion. The first discriminator uses spectral norm
                and the other discriminators use weight norm.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params["use_weight_norm"] = False
                    params["use_spectral_norm"] = True
                else:
                    params["use_weight_norm"] = True
                    params["use_spectral_norm"] = False
            self.discriminators += [DDSPScaleDiscriminator(**params)]
        self.pooling = None
        if scales > 1:
            self.pooling = getattr(torch.nn, downsample_pooling)(
                **downsample_pooling_params
            )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[List[torch.Tensor]]: List of list of each discriminator outputs,
                which consists of eachlayer output tensors.

        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            if self.pooling is not None:
                x = self.pooling(x)

        return outs

class PitchDiscriminator(torch.nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(PitchDiscriminator, self).__init__()
        self.hidden_size = channels
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(channels),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm1d(channels),
            torch.nn.LeakyReLU(0.2),
            
            torch.nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm1d(channels),
            torch.nn.LeakyReLU(0.2),
        )

        # self.fc = torch.nn.Sequential(
        #     torch.nn.Linear(channels, out_channels),
        #     torch.nn.Sigmoid()
        # )

    def forward(self, x):
        batch_size = x.size(0)
        max_length = x.size(1)
        # print("x",x.shape)
        # print("x1",x.unsqueeze(1).shape)
        x = self.cnn(x.unsqueeze(1))
        x = x.view(batch_size, self.hidden_size, -1)
        # print("x2",x.shape)
        # x = self.fc(x)
        # x = x.squeeze()
        x = [x]
        return x

class DDSPMultiScaleMultiPeriodPitchDiscriminator(torch.nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-scale discriminator related
        scales: int = 3,
        scale_downsample_pooling: str = "AvgPool1d",
        scale_downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm: bool = True,
        # Multi-period discriminator related
        periods: List[int] = [2, 3, 5, 7, 11],
        period_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
        pitch_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "channels": 64,
            "out_channels": 1,
        },
    ):
        """Initilize DDSP multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling
                module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale
                discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the
                official implementaion. The first discriminator uses spectral norm and
                the other discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period
                discriminator module. The period parameter will be overwritten.

        """
        super().__init__()
        self.msd = DDSPMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm,
        )
        self.mpd = DDSPMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )
        params = copy.deepcopy(pitch_discriminator_params)
        self.pd = PitchDiscriminator(**params)

    def forward(self, x: torch.Tensor, pitch: torch.Tensor) -> List[List[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[List[Tensor]]: List of list of each discriminator outputs,
                which consists of each layer output tensors. Multi scale and
                multi period ones are concatenated.

        """
        msd_outs = self.msd(x)
        # print("msd_outs", msd_outs)
        # print("msd len", len(msd_outs))
        # print("msd len", len(msd_outs[0]))
        # print("msd_outs shape", msd_outs[0][0].shape)
        mpd_outs = self.mpd(x)
        # print("mpd len", len(msd_outs))
        # print("mpd_outs", mpd_outs)
        # print("mpd_outs shape", mpd_outs[0][0].shape)
        # raise ValueError
        # print("out shape",len(msd_outs + mpd_outs)) # 6
        # print("out shape",len((msd_outs + mpd_outs)[0])) # 8
        pd_outs = self.pd(pitch)
        return msd_outs + mpd_outs + pd_outs

class DDSPMultiScaleMultiPeriodDiscriminator(torch.nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-scale discriminator related
        scales: int = 3,
        scale_downsample_pooling: str = "AvgPool1d",
        scale_downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm: bool = True,
        # Multi-period discriminator related
        periods: List[int] = [2, 3, 5, 7, 11],
        period_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initilize DDSP multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling
                module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale
                discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the
                official implementaion. The first discriminator uses spectral norm and
                the other discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period
                discriminator module. The period parameter will be overwritten.

        """
        super().__init__()
        self.msd = DDSPMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm,
        )
        self.mpd = DDSPMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

    def forward(self, x: torch.Tensor, pitch: torch.Tensor) -> List[List[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[List[Tensor]]: List of list of each discriminator outputs,
                which consists of each layer output tensors. Multi scale and
                multi period ones are concatenated.

        """
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        return msd_outs + mpd_outs

class MultiFrequencyDiscriminator(torch.nn.Module):
    def __init__(self,
                 hop_lengths=[128, 256, 512],
                 hidden_channels=[256, 512, 512],
                 domain='double', mel_scale=True):
        super(MultiFrequencyDiscriminator, self).__init__()

        self.stfts = torch.nn.ModuleList([
            TorchSTFT(fft_size=x * 4, hop_size=x, win_size=x * 4,
                      normalized=True, domain=domain, mel_scale=mel_scale)
            for x in hop_lengths])

        self.domain = domain
        if domain == 'double':
            self.discriminators = torch.nn.ModuleList([
                BaseFrequenceDiscriminator(2, c)
                for x, c in zip(hop_lengths, hidden_channels)])
        else:
            self.discriminators = torch.nn.ModuleList([
                BaseFrequenceDiscriminator(1, c)
                for x, c in zip(hop_lengths, hidden_channels)])

    def forward(self, x):
        scores, feats = list(), list()
        for stft, layer in zip(self.stfts, self.discriminators):
            # print(stft)
            mag, phase = stft.transform(x.squeeze())
            if self.domain == 'double':
                mag = torch.stack(torch.chunk(mag, 2, dim=1), dim=1)
            else:
                mag = mag.unsqueeze(1)

            score, feat = layer(mag)
            scores.append(score)
            feats.append(feat)
        return scores, feats

class BaseFrequenceDiscriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super(BaseFrequenceDiscriminator, self).__init__()

        self.discriminator = torch.nn.ModuleList()
        self.discriminator += [
            torch.nn.Sequential(
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(torch.nn.Conv2d(
                    in_channels, hidden_channels // 32,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(torch.nn.Conv2d(
                    hidden_channels // 32, hidden_channels // 16,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(torch.nn.Conv2d(
                    hidden_channels // 16, hidden_channels // 8,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(torch.nn.Conv2d(
                    hidden_channels // 8, hidden_channels // 4,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(torch.nn.Conv2d(
                    hidden_channels // 4, hidden_channels // 2,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(torch.nn.Conv2d(
                    hidden_channels // 2, hidden_channels,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(torch.nn.Conv2d(
                    hidden_channels, 1,
                    kernel_size=(3, 3), stride=(1, 1)))
            )
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[-1]