import cv2
import numpy
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchinfo import summary

from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.metrics import structural_similarity
import lpips

import matplotlib.pyplot as plt

VALID_DATASET_PATH = "./dataset/DIV2K_valid_HR/"

LOW_RESOLUTION_WIDTH = 32
LOW_RESOLUTION_HEIGHT = 32
DATASET_LOW_RESOLUTION_PATH = "./dataset/32_images/"
DATASET_HIGH_RESOLUTION_PATH = "./dataset/256_images/"

TRAIN_NUMBER_OF_IMAGES = 800
VALID_NUMBER_OF_IMAGES = 100

TRAIN_INDEXES = (1, TRAIN_NUMBER_OF_IMAGES)
VALID_INDEXES = (TRAIN_NUMBER_OF_IMAGES + 1, TRAIN_NUMBER_OF_IMAGES + VALID_NUMBER_OF_IMAGES)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


class SuperResolutionDataset(Dataset):
  def __init__(self, path_low_resolution, path_high_resolution, train_sizes, transform=None):
    self.path_low = path_low_resolution
    self.sizes = train_sizes
    self.path_high = path_high_resolution
    self.transform = transform


  def __len__(self):
    return self.sizes[1] - self.sizes[0] + 1

  def __getitem__(self, idx):
    number = str(idx + self.sizes[0])
    number_len = len(number)
    number = (4 - number_len) * "0" + number

    filename_path_low = self.path_low + number + ".png"
    filename_path_high = self.path_high + number + ".png"

    cv_img_low = cv2.imread(filename_path_low)
    cv_img_high = cv2.imread(filename_path_high)

    image_low  = cv2.cvtColor(cv_img_low, cv2.COLOR_BGR2RGB)
    image_high = cv2.cvtColor(cv_img_high, cv2.COLOR_BGR2RGB)

    if self.transform:
        image_low = self.transform(image_low)
        image_high = self.transform(image_high)
    return image_low, image_high
  
transform = transforms.ToTensor()

valid_dataset = SuperResolutionDataset(DATASET_LOW_RESOLUTION_PATH + "valid/", DATASET_HIGH_RESOLUTION_PATH + "valid/", VALID_INDEXES, transform=transform)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, pin_memory=True)



class ResBlock(torch.nn.Module):
  def __init__(self, in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1,
               bias=True):
    super(ResBlock, self).__init__()
    layers = [torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias),
              torch.nn.ReLU(inplace=True),
              torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                        padding=padding, bias=bias)]
    self.body = torch.nn.Sequential(*layers)
    self.res_scale = 0.1

  def forward(self, x):
    res = self.body(x).mul(self.res_scale)
    res += x
    return res


class Upsampler(torch.nn.Sequential):
  def __init__(self, channels = 32, low_resolution_size = 64):
    layers = []
    if low_resolution_size == 32:
      layers.append(torch.nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=True))
      layers.append(torch.nn.PixelShuffle(2))
      layers.append(torch.nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=True))
      layers.append(torch.nn.PixelShuffle(2))
      layers.append(torch.nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=True))
      layers.append(torch.nn.PixelShuffle(2))
    elif low_resolution_size == 64:
      layers.append(torch.nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=True))
      layers.append(torch.nn.PixelShuffle(2))
      layers.append(torch.nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=True))
      layers.append(torch.nn.PixelShuffle(2))
    else:
      raise NotImplementedError
    
    super(Upsampler, self).__init__(*layers)


class EDSR(torch.nn.Module):
  def __init__(self, channels = 32, low_resolution_size = 64, body_layers_num = 12):
    super(EDSR, self).__init__()

    head_layers = [torch.nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)]
    body_layers = []

    for _ in range(body_layers_num):
      body_layers.append(ResBlock(in_channel = channels, out_channel = channels))
    body_layers.append(torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True))
    
    tail_layers = [Upsampler(channels = channels, low_resolution_size = low_resolution_size),
                   torch.nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)]
    
    self.head = torch.nn.Sequential(*head_layers)
    self.body = torch.nn.Sequential(*body_layers)
    self.tail = torch.nn.Sequential(*tail_layers)

  def forward(self, x):
    x = self.head(x)
    res = self.body(x)
    res += x
    x = self.tail(res)
    return x


model = EDSR(channels = 64, body_layers_num = 14, low_resolution_size = LOW_RESOLUTION_WIDTH)
model.load_state_dict(torch.load("super_resolution_model32_2.pt", weights_only=True))
model.eval()
model.to(device)
summary(model, input_size=(1, 3, 32, 32))

if not os.path.exists(f"{DATASET_HIGH_RESOLUTION_PATH}generated/"):
    os.makedirs(f"{DATASET_HIGH_RESOLUTION_PATH}generated/")

with torch.no_grad():
  for i,data in enumerate(valid_loader):
    image_lr, image_hr = data
    image_lr = image_lr.to(device)
    img_tensors = model(image_lr).cpu()
    for j, img_tensor in enumerate(torch.split(img_tensors, 1)):
      img_tensor = img_tensor.squeeze(0)
      img_numpy = img_tensor.numpy()
      cv2_image = numpy.transpose(img_numpy, (1, 2, 0))
      cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR) * 255
      cv2.imwrite(f"{DATASET_HIGH_RESOLUTION_PATH}generated/0{801+i*16+j}.png", cv2_image)



sne_model = []
sne_baseline = []

psnr_model = []
psnr_baseline = []

ssim_model = []
ssim_baseline = []

lpips_model = []
lpips_baseline = []
loss_fn_alex = lpips.LPIPS(net='alex')

for i in range(1, 101):
  number = str(i + 800)
  number_len = len(number)
  number = (4 - number_len) * "0" + number

  filename_path_low = DATASET_LOW_RESOLUTION_PATH + "valid/" + number + ".png"
  filename_path_high = DATASET_HIGH_RESOLUTION_PATH + "valid/" + number + ".png"
  filename_path_generated = DATASET_HIGH_RESOLUTION_PATH + "generated/" + number + ".png"


  image_org = cv2.imread(filename_path_high, cv2.COLOR_BGR2RGB)
  image_lr =  cv2.imread(filename_path_low, cv2.COLOR_BGR2RGB)
  image_generated = cv2.imread(filename_path_generated, cv2.COLOR_BGR2RGB)
  image_baseline = cv2.resize(image_lr, (256, 256), interpolation = cv2.INTER_CUBIC)

  sne_model.append(mean_squared_error(image_org, image_generated))
  sne_baseline.append(mean_squared_error(image_org, image_baseline))

  psnr_model.append(peak_signal_noise_ratio(image_org, image_generated))
  psnr_baseline.append(peak_signal_noise_ratio(image_org, image_baseline))

  ssim_model.append(structural_similarity(image_org, image_generated, channel_axis=-1))
  ssim_baseline.append(structural_similarity(image_org, image_baseline, channel_axis=-1))

  image_org_tensor = lpips.im2tensor(image_org)
  image_generated_tensor = lpips.im2tensor(image_generated)
  image_baseline_tensor = lpips.im2tensor(image_baseline)
  lpips_model.append(loss_fn_alex(image_org_tensor, image_generated_tensor))
  lpips_baseline.append(loss_fn_alex(image_org_tensor, image_baseline_tensor))


print(f"SNE for bilateral denoising: {sum(sne_baseline)/len(sne_baseline):.3f}")
print(f"SNE for model: {sum(sne_model)/len(sne_model):.3f}")

print(f"PSNR for bilateral denoising: {sum(psnr_baseline)/len(psnr_baseline):.3f}")
print(f"PSNR for model: {sum(psnr_model)/len(psnr_model):.3f}")

print(f"SSIM for bilateral denoising: {sum(ssim_baseline)/len(ssim_baseline):.3f}")
print(f"SSIM for model: {sum(ssim_model)/len(ssim_model):.3f}")

print(f"LPIPS for bilateral denoising: {(sum(lpips_baseline)/len(lpips_baseline)).detach().item():.3f}")
print(f"LPIPS for model: {(sum(lpips_model)/len(lpips_model)).detach().item():.3f}")