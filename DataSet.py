import torch.utils.data as data
from os import listdir
from os.path import join
import cv2
import copy
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as tf
class DataSetFromFolder(data.Dataset):
    def __init__(self, image_dir, target_dir, if_test, pae):
        super(DataSetFromFolder, self).__init__()
        self.pae = pae
        self.image_filenames = [x for x in listdir(target_dir)]
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.if_test = if_test
        # self.target_filenames = [join(target_dir, x) for x in listdir(target_dir)]

    def transform(self, image, label):
        if random.random() > 0.8:
            angle = transforms.RandomRotation.get_params([-180,180])
            image = tf.rotate(image, angle)
            label = tf.rotate(label, angle)
        return image, label

    def __getitem__(self, index):
        data_dir_element = np.random.choice([2,3,4,5,6,7,8,9,10])
        if self.if_test:
            data_dir_element = self.pae
            input_image = cv2.imread(join(self.image_dir+str(data_dir_element), self.image_filenames[index]), 0)
            target_image = cv2.imread(join(self.target_dir, self.image_filenames[index]), 0)
        else:

            tmp_dir = copy.copy(self.image_dir)
            tmp_dir = join(tmp_dir, str(data_dir_element) + '/')
            input_image = cv2.imread(join(tmp_dir, self.image_filenames[index]), 0)
            target_image = cv2.imread(join(self.target_dir, self.image_filenames[index]), 0)

        # input_image = (input_image / tau -21.5) / 21.5
        # target_image = (target_image / tau -21.5) / 21.5
        # self.data_dir_element.astype(np.float32)
        input_image = (input_image  - 128.0) / 128.0 / (0.7 * data_dir_element)
        target_image = (target_image  - 128.0) / 128.0 / (0.7 * data_dir_element)

        input_image = Image.fromarray(input_image)
        target_image = Image.fromarray(target_image)
    
        if not self.if_test:
            input_image, target_image = self.transform(input_image, target_image)

        input_image = transforms.ToTensor()(input_image)
        target_image = transforms.ToTensor()(target_image)
        return input_image, target_image

    def __len__(self):
        return int(len(self.image_filenames))
        # return 10000
