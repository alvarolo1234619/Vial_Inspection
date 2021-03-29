# Copyright 2020 NVIDIA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    device = torch.device('cuda')
    image= image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    # image = PIL.Image.fromarray(image) # Transforms image into a PIL image
    image=image.to(device)
    image = transforms.functional.resize(image, (224,224))
    # image = np.array(image)
    # image = transforms.functional.to_tensor(image)  #.to(device)
    image=image.float()
    
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# image=cv2.imread("top_view_vials/airnob/airnob10.jpg") This did not Work!
# image=cv2.imread("/home/alvaro/Desktop/pyPro/opencv_program/top_view_vials/airnob/airnob10.jpg") 

# cv2.imshow("MyImage",image)
# cv2.waitKey(0)
# preprocess(cv2.imread("test.jpg"))
# print(mean.is_cuda )
# print(std.is_cuda)
# test_img = cv2.imread("test.jpg")
# print(preprocess(test_img) )
# test_img= torch.from_numpy(test_img)
# test_img=test_img.to("cuda")
# print(test_img.is_cuda)
def tensor2plt(image, normalize=True):
    """Imshow for Tensor."""
    # if ax is None:
    #     fig, ax = plt.subplots()
    image = image.to("cpu").numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    # ax.imshow(image)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.tick_params(axis='both', length=0)
    # ax.set_xticklabels('')
    # ax.set_yticklabels('')

    return image