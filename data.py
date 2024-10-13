from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode:str):
        super().__init__()
        self.data = data
        self.mode = mode
        self._transform = self.get_transform()

    def get_transform(self):
        if self.mode == "train":
            random_angle = np.random.randint(0, 90)
            return tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                #tv.transforms.RandomResizedCrop(224),  # Randomly crop the image and resize it to 224x224
                #tv.transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                #tv.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                #tv.transforms.GaussianBlur(kernel_size=3),

                # Randomly adjust brightness, contrast, saturation, and hue
                tv.transforms.RandomHorizontalFlip(p=0.4), ###################### BEST results were for p=0.4
                tv.transforms.RandomVerticalFlip(p=0.4), #### Tried p=0.2 but gave overfitting after some time
                #tv.transforms.RandomRotation(20),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            return tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, crack, inactive = self.data.iloc[index,:]
        gray_img = imread(img_path) # gray img  from path
        rgb_img = gray2rgb(gray_img)  # grayscale to RGB
        rgb_img_transformed = self._transform(rgb_img)# img transformed according to train or val, case handled in
        label_tensor = torch.tensor([crack, inactive], dtype=torch.float32)
        return rgb_img_transformed, label_tensor

#
# from torch.utils.data import Dataset
# import torch
# from pathlib import Path
# from skimage.io import imread
# from skimage.color import gray2rgb
# import numpy as np
# import torchvision as tv
# from scipy.ndimage import map_coordinates
# from scipy.ndimage.filters import gaussian_filter
#
# train_mean = [0.59685254, 0.59685254, 0.59685254]
# train_std = [0.16043035, 0.16043035, 0.16043035]
#
# class ElasticTransform:
#     def __init__(self, alpha, sigma):
#         self.alpha = alpha
#         self.sigma = sigma
#
#     def __call__(self, img):
#         random_state = np.random.RandomState(None)
#         img = np.array(img)
#         shape = img.shape
#         dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
#         dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
#
#         x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
#         indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
#
#         distored_image = map_coordinates(img, indices, order=1, mode='reflect')
#         return distored_image.reshape(img.shape)
#
# # class Cutout:
# #     def __init__(self, size):
# #         self.size = size
# #
# #     def __call__(self, img):
# #         img = np.array(img)
# #         h, w = img.shape[:2]
# #         mask = np.ones((h, w), np.float32)
# #         y = np.random.randint(h)
# #         x = np.random.randint(w)
# #
# #         y1 = np.clip(y - self.size // 2, 0, h)
# #         y2 = np.clip(y + self.size // 2, 0, h)
# #         x1 = np.clip(x - self.size // 2, 0, w)
# #         x2 = np.clip(x + self.size // 2, 0, w)
# #
# #         mask[y1:y2, x1:x2] = 0
# #         img = img * mask[:, :, None]
# #         return img
#
# class ChallengeDataset(Dataset):
#     def __init__(self, data, mode: str):
#         super().__init__()
#         self.data = data
#         self.mode = mode
#         self._transform = self.get_transform()
#
#     def get_transform(self):
#         if self.mode == "train":
#             return tv.transforms.Compose([
#                 tv.transforms.ToPILImage(),
#                 tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#                 tv.transforms.RandomHorizontalFlip(p=0.4),
#                 tv.transforms.RandomVerticalFlip(p=0.4),
#                 ElasticTransform(alpha=100, sigma=10),
#                 #Cutout(size=40),
#                 tv.transforms.ToTensor(),
#                 tv.transforms.Normalize(mean=train_mean, std=train_std)
#             ])
#         else:
#             return tv.transforms.Compose([
#                 tv.transforms.ToPILImage(),
#                 tv.transforms.ToTensor(),
#                 tv.transforms.Normalize(mean=train_mean, std=train_std)
#             ])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         img_path, crack, inactive = self.data.iloc[index, :]
#         gray_img = imread(img_path)
#         rgb_img = gray2rgb(gray_img)
#         rgb_img_transformed = self._transform(rgb_img)
#         label_tensor = torch.tensor([crack, inactive], dtype=torch.float32)
#         return rgb_img_transformed, label_tensor
