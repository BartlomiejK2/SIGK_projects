import cv2
import numpy
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchinfo import summary

import matplotlib.pyplot as plt

TRAIN_DATASET_PATH = "./dataset/DIV2K_train_HR/"
VALID_DATASET_PATH = "./dataset/DIV2K_valid_HR/"

LOW_RESOLUTION_WIDTH = 32
LOW_RESOLUTION_HEIGHT = 32
DATASET_LOW_RESOLUTION_PATH = "./dataset/32_images/"
DATASET_HIGH_RESOLUTION_PATH = "./dataset/256_images/"
TRAIN_NUMBER_OF_IMAGES = 800
VALID_NUMBER_OF_IMAGES = 100

TRAIN_INDEXES = (1, TRAIN_NUMBER_OF_IMAGES)
VALID_INDEXES = (TRAIN_NUMBER_OF_IMAGES + 1, TRAIN_NUMBER_OF_IMAGES + VALID_NUMBER_OF_IMAGES)


def generate_resized_images(path: str, images_num: tuple, width: int, height: int, output_path: str):
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  for i in range(images_num[0], images_num[1] + 1):
    number = str(i)
    number_len = len(number)
    number = (4 - number_len) * "0" + number
    filename_path = path + number + ".png"
    orginal_image = cv2.imread(filename_path)
    points = (width, height)
    resized_image = cv2.resize(orginal_image, points, interpolation= cv2.INTER_CUBIC)
    new_filename_path = output_path + number + ".png"
    cv2.imwrite(new_filename_path, resized_image)

# # # Obrazy dla zbioru trenującego
# generate_resized_images(TRAIN_DATASET_PATH, TRAIN_INDEXES, LOW_RESOLUTION_WIDTH, LOW_RESOLUTION_HEIGHT, DATASET_LOW_RESOLUTION_PATH + "train/")
# generate_resized_images(TRAIN_DATASET_PATH, TRAIN_INDEXES, 256, 256, DATASET_HIGH_RESOLUTION_PATH + "train/")

# # Obrazy dla zbioru walidacyjnego
# generate_resized_images(VALID_DATASET_PATH, VALID_INDEXES, LOW_RESOLUTION_WIDTH, LOW_RESOLUTION_HEIGHT, DATASET_LOW_RESOLUTION_PATH + "valid/")
# generate_resized_images(VALID_DATASET_PATH, VALID_INDEXES, 256, 256, DATASET_HIGH_RESOLUTION_PATH + "valid/")


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

    cv_img_low = cv2.imread(filename_path_low, cv2.COLOR_BGR2RGB)
    cv_img_high = cv2.imread(filename_path_high, cv2.COLOR_BGR2RGB)

    if self.transform:
        image_low = self.transform(cv_img_low)
        image_high = self.transform(cv_img_high)
    return image_low, image_high


transform = transforms.ToTensor()

train_dataset = SuperResolutionDataset(DATASET_LOW_RESOLUTION_PATH + "train/", DATASET_HIGH_RESOLUTION_PATH + "train/", TRAIN_INDEXES, transform=transform)
valid_dataset = SuperResolutionDataset(DATASET_LOW_RESOLUTION_PATH + "valid/", DATASET_HIGH_RESOLUTION_PATH + "valid/", VALID_INDEXES, transform=transform)

print(train_dataset[10][0])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, pin_memory=True)


# Based on https://github.com/wangzhesun/super_resolution/tree/main

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

summary(model, input_size=(1, 3, 32, 32))



criterion = torch.nn.L1Loss(size_average=False)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 15
epoch_vec = [i for i in range(epochs)]
for epoch in range(epochs):
  running_loss = 0.0
  for data in train_loader:
    image_lr, image_hr = data
    image_lr, image_hr = image_lr.to(device), image_hr.to(device)
    optimizer.zero_grad()
    outputs = model(image_lr)
    loss = criterion(outputs, image_hr) / (image_lr.size()[0])
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  loss = running_loss / len(train_loader)
  print('Epoch {} of {}, Train Loss: {:.4f}'.format(epoch+1, epochs, loss))


torch.save(model.state_dict(), "super_resolution_model32_2.pt")
