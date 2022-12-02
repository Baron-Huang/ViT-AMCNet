############################# Grad-CAM.py ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: A python script for realizing a Gram_CAM methods

########################## API Section #########################
import cv2
import os
import numpy as np
import skimage
import skimage.data
import skimage.io
import skimage.transform
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

def read_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''

    img = Image.open(picture_dir)
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "1468grad_cam.jpg")
    path_raw_img = os.path.join(out_dir, "1468original.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 4).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (200, 200))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':

    pic_dir = '/home/zxl/数据集/脑癌/sequence_crop_4classes/test/IV/1468.jpg'
    path_net = '/home/zxl/训练权重/pytorch/BrainNet/Base_Models/resnet50.pth'
    output_dir = '/home/zxl/图片/ASI-DBNet/stage5_fms/IV/IV_resnet50_output/'

    classes = ('I', 'II', 'III', 'IV')
    fmap_block = list()
    grad_block = list()

    # 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 图片读取；网络加载
    img_input = read_picture(pic_dir, transform)    #img_input.shape=torch.Size([1, 3, 224, 224])

    #创建和加载模型
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(in_features=2048, out_features=4, bias=True)
    #net.load_state_dict(torch.load(path_net))

    # 注册hook
    net.layer4[2].conv3.register_forward_hook(farward_hook)
    net.layer4[2].conv3.register_backward_hook(backward_hook)

    # forward
    output = net(img_input)         #output.shape=torch.Size([1, 4])
    idx = np.argmax(output.cpu().data.numpy())      #idx=3,是该图片预测类别的位置索引
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()      #grad_block[0]:特征图对应的梯度,
    # grad_block[0].shape=torch.Size([1, 2048, 7, 7]),和特征图尺寸一样，grads_val.shape=torch.Size([2048, 7, 7])
    fmap = fmap_block[0].cpu().data.numpy().squeeze()           #fmap_block[0]：所要获取的特征图
    # fmap_block[0].shape=torch.Size([1, 2048, 7, 7]),和特征图尺寸一样，fmap.shape=torch.Size([2048, 7, 7])
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img = cv2.imread(pic_dir, 1)        #img.shape=(200, 200, 3)
    img_show = np.float32(cv2.resize(img, (200, 200))) / 255      #img_show.shape=(32, 32, 3)
    show_cam_on_image(img_show, cam, output_dir)
