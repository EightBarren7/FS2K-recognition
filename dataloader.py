import json
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


def dataloader(json_path, data_dir, batch_size, shuffle=True):
    fp = open(json_path, 'r')
    file = json.load(fp)

    image_path_list = []
    label_list = []
    for data in file:
        per_label = []
        image_path_list.append('sketch' + data['image_name'][5:7] + 'sketch' + data['image_name'][12:])
        per_label.append(data['hair'])
        per_label.append(data['hair_color'])
        per_label.append(data['gender'])
        per_label.append(data['earring'])
        per_label.append(data['smile'])
        per_label.append(data['frontal_face'])
        per_label.append(data['style'])
        label_list.append(torch.tensor(per_label))

    image_list = []

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for image_path in image_path_list:
        if 'sketch2/' in image_path:
            data_path = os.path.join(data_dir, image_path + '.png')
        else:
            data_path = os.path.join(data_dir, image_path + '.jpg')
        img = Image.open(data_path).convert('RGB')
        img_tensor = data_transforms(img)
        image_list.append(img_tensor)

    iter_list = []
    for i in range(len(image_list)):
        iter_list.append((image_list[i], label_list[i]))

    data_iter = DataLoader(iter_list, batch_size=batch_size, shuffle=shuffle)
    return data_iter
