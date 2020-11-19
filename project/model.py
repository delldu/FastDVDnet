"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:51:08 CST
# ***
# ************************************************************************************/
#

import math
import os
import pdb
import sys

import torch
import torch.nn as nn
from apex import amp
from tqdm import tqdm

from data import VIDEO_SEQUENCE_LENGTH


def PSNR(img1, img2):
    """PSNR."""
    difference = (1.*img1-img2)**2
    mse = torch.sqrt(torch.mean(difference)) + 0.000001
    return 20*torch.log10(1./mse)

# The following comes from
# https://github.com/m-tassano/fastdvdnet
# Thanks a lot.


class ConvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputConvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch):
        super(InputConvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames * (3 + 1),
                      num_in_frames * self.interm_ch,
                      kernel_size=3,
                      padding=1,
                      groups=num_in_frames),
            nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames * self.interm_ch,
                      out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ConvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            ConvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputConvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch):
        super(OutputConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.convblock(x)


class DenoiseBlock(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=3):
        super(DenoiseBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputConvBlock(
            num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputConvBlock(in_ch=self.chs_lyr0, out_ch=3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        x0 = self.inc(
            torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        # Downsampling
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        # Upsampling
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        # Estimation
        x = self.outc(x0 + x1)

        # Residual
        x = in1 - x

        return x


class VideoCleanModel(nn.Module):
    """VideoClean Model."""

    def __init__(self, num_input_frames=5):
        '''Init model.'''
        super(VideoCleanModel, self).__init__()

        self.num_input_frames = num_input_frames
        # Define models of each denoising stage
        self.temp1 = DenoiseBlock(num_input_frames=3)
        self.temp2 = DenoiseBlock(num_input_frames=3)
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):
        '''Forward.
        Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Unpack inputs

        (x0, x1, x2, x3, x4) = tuple(x[:, 3 * m:3 * m + 3, :, :]
                                     for m in range(self.num_input_frames))
        # First stage
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)

        # Second stage
        x = self.temp2(x20, x21, x22, noise_map)

        return x

    # def __init__(self):
    #     """Init model."""
    #     super(VideoCleanModel, self).__init__()

    # def forward(self, x):
    #     """Forward."""
    #     return x


def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)


def export_onnx_model():
    """Export onnx model."""

    import onnx
    from onnx import optimizer

    onnx_file = "output/video_clean.onnx"
    weight_file = "output/VideoClean.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)

    input_names = ["input"]
    output_names = ["output"]
    # variable lenght axes
    dynamic_axes = {'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},
                    'output': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'}}
    torch.onnx.export(model, dummy_input, onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True,
                      export_params=True,
                      dynamic_axes=dynamic_axes)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer",
              "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('image_clean.onnx')"


def export_torch_model():
    """Export torch model."""

    script_file = "output/video_clean.pt"
    weight_file = "output/VideoClean.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(script_file)


def get_model():
    """Create model."""
    model_setenv()
    model = VideoCleanModel(num_input_frames=5)
    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()
    model.train()
    criterion = nn.MSELoss(reduction='sum')
    start_channel = 3 * (VIDEO_SEQUENCE_LENGTH//2)
    stop_channel = start_channel + 3

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images = data
            count = len(images)

            # noise
            # input_tensor = images + noise
            # noise_tensor = ...

            # output_tensor = model(input_tensor, noise_tensor)
            # loss = criterion(images, output_tensor)
            N, C, H, W = images.size()
            stdn = torch.empty((N, 1, 1, 1)).uniform_(5.0, 55.0)/255.0
            noise = torch.zeros_like(images)
            noise = torch.normal(mean=noise, std=stdn.expand_as(noise))

            input_tensor = images + noise
            noise_tensor = stdn.expand((N, 1, H, W))  # one channel per image

            # images = images.to(device)
            input_tensor = input_tensor.to(device)
            noise_tensor = noise_tensor.to(device)
            GT = images[:, start_channel: stop_channel, :, :].to(device)

            output_tensor = model(input_tensor, noise_tensor)
            # pdb.set_trace()

            loss = criterion(GT, output_tensor)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()
    start_channel = 3 * (VIDEO_SEQUENCE_LENGTH//2)
    stop_channel = start_channel + 3

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images = data
            count = len(images)

            N, C, H, W = images.size()

            # Transform data to device
            GT = images[:, start_channel: stop_channel, :, :].to(device)

            noise = torch.FloatTensor(images.size()).normal_(
                mean=0, std=25.0/255.0)
            input_tensor = images + noise
            input_tensor = input_tensor.to(device)

            noise_std = torch.FloatTensor([25.0/255.0])
            noise_tensor = noise_std.expand((N, 1, H, W))
            noise_tensor = noise_tensor.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(input_tensor, noise_tensor)

            loss_value = PSNR(predicts, GT)
            valid_loss.update(loss_value, count)
            t.set_postfix(loss='PSNR:{:.6f}'.format(valid_loss.avg))
            t.update(count)


def model_device():
    """First call model_setenv. """
    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default environment variables to avoid exceptions
    if os.environ.get("ONLY_USE_CPU") != "YES" and os.environ.get("ONLY_USE_CPU") != "NO":
        os.environ["ONLY_USE_CPU"] = "NO"

    if os.environ.get("ENABLE_APEX") != "YES" and os.environ.get("ENABLE_APEX") != "NO":
        os.environ["ENABLE_APEX"] = "YES"

    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else:
        os.environ["ENABLE_APEX"] = "YES"

    # Running on GPU if available
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def enable_amp(x):
    """Init Automatic Mixed Precision(AMP)."""
    if os.environ["ENABLE_APEX"] == "YES":
        x = amp.initialize(x, opt_level="O1")


def infer_perform():
    """Model infer performance ..."""

    model = get_model()
    model.eval()
    device = model_device()
    model = model.to(device)

    progress_bar = tqdm(total=100)
    progress_bar.set_description("Test Inference Performance ...")

    for i in range(100):
        input = torch.randn(8, 3, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input)

        progress_bar.update(1)


if __name__ == '__main__':
    """Test model ..."""

    model = get_model()
    print(model)

    export_torch_model()
    export_onnx_model()

    infer_perform()
