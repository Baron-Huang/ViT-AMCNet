'''
###Funcition:Main fucntion of manuscript:'A ViT-AMC Network with Adaptive Model Fusion and Multi-Objective
             Optimization for Interpretable Laryngeal Tumor Grading from Histopathological Images'.
###Author: Dr.Pan Huang
###E-mail: panhuang@cqu.edu.cn
###Creating Data:2021.11.29
###Department: College of Optoelectronic Engineering, Chongqing University, China.
'''

import torch
from torch import nn
from torchvision.models.densenet import densenet121
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torchsummaryX import summary
from tensorboardX import SummaryWriter
from DDAI_Utils.ViT import VisionTransformer, load_pretrained, vit_base_patch16_224, vit_small_patch16_224,\
    vit_base_patch32_224, vit_large_patch16_224, vit_large_patch32_224
from DDAI_Utils.model_modules import DDAI_TCNet_Ablation, Dense_Net, FABNet, ViT_Net, DDAI_TCNet, ViT_Large_Net
from DDAI_Utils.fit_functions import Double_out_fit, vit_lr_schedule, Single_out_fit, Triplet_out_fit
from DDAI_Utils.ablation_experiments import save_model, acc_scores
from DDAI_Utils.vit_grad_rollout import VITAttentionGradRollout
from DDAI_Utils.vit_rollout import VITAttentionRollout, show_mask_on_image
import cv2
from skimage import io

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    random.seed(7 + worker_id)
    np.random.seed(7 + worker_id)
    torch.manual_seed(7 + worker_id)
    torch.cuda.manual_seed(7 + worker_id)
    torch.cuda.manual_seed_all(7 + worker_id)

if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(1234)
    #write_1 = SummaryWriter('/home/zhouxiaoli/DDAI-TCNet/Log/Dense_SE')
    gpu_device = 0
    class_num = 3
    # work_mode = {'normal', 'featrues_extraction'}
    work_mode = 'normal'
    # train_mode = {'MLMT', 'FDAI', 'FDAF', 'Featrues_stacking', 'Probability_fusion'}}
    train_mode = 'FDAF'
    # model_mode = {'MLMT', 'FDAI', 'FDAF', 'Featrues_stacking', 'Probability_fusion'}
    model_mode = 'FDAF'
    # 预处理 转为tensor 以及 标准化
    transform = transforms.Compose([transforms.Resize([224, 224]) ,transforms.ToTensor()
                                   , transforms.Normalize(mean=0.5, std=0.5)])

    # 使用torchvision.datasets.ImageFolder读取数据集 指定train 和 test文件夹
    train_data = ImageFolder(r'E:\DDAI-TCNet-Github\Datasets\Larynx_dataset\Train_patch', transform=transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1)

    test_data = ImageFolder(r'E:\DDAI-TCNet-Github\Datasets\Larynx_dataset\Test_patch', transform=transform)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=1)

    val_data = ImageFolder(r'E:\DDAI-TCNet-Github\Datasets\Larynx_dataset\Val_patch', transform=transform)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=1)


    ###Single FABNet experiments
    '''
    dense_base = densenet121(pretrained=False)
    dense_weight_file = torch.load(r'E:\DDAI-TCNet-Github\Weights\densenet121-a639ec97.pth')
    dense_base.load_state_dict(dense_weight_file, strict=False)

    fab_net = FABNet(base_net=dense_base, class_num=3)
    print(summary(fab_net, torch.randn((1, 3, 224, 224))))
    print(fab_net)
    fab_net = fab_net.cuda(gpu_device)
    '''

    ###Single ViT experiments
    '''
    vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=3, embed_dim=768,
                depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    vit_weight_file = torch.load(r'E:\DDAI-TCNet-Github\Weights\jx_vit_base_p16_224-80ecf9dd.pth')
    vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    vit_base.load_state_dict(vit_weight_file, strict=False)
    vit_net = ViT_Net(base_net=vit_base, class_num=3)
    print(summary(vit_net , torch.randn((1, 3, 224, 224))))
    print(vit_net)
    vit_net  = vit_net.cuda(gpu_device)
    '''

    ###Single ViT_x ablation experiments
    '''
    vit_base = vit_base = VisionTransformer(img_size=224, patch_size=32, in_chans=3, num_classes=3, embed_dim=1024,
                depth=24, num_heads=16, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)

    vit_weight_file = torch.load('/home/zhouxiaoli/DDAI-TCNet/Weights/jx_vit_large_patch32_224_in21k-9046d2e7.pth')
    vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    #vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'patch_embed.proj' not in k)}
    vit_base.load_state_dict(vit_weight_file, strict=False)
    vit_net = ViT_Large_Net(base_net=vit_base, class_num=3)
    print(summary(vit_net , torch.randn((1, 3, 224, 224))))
    print(vit_net)
    vit_net  = vit_net.cuda(gpu_device)
    '''

    ###ViT-AMCNet_Ablation experiments
    '''
    vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=3, embed_dim=768,
                                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    vit_weight_file = torch.load('/home/zhouxiaoli/DDAI-TCNet/Weights/jx_vit_base_p16_224-80ecf9dd.pth')
    vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    vit_base.load_state_dict(vit_weight_file, strict=False)

    dense_base = densenet121(pretrained=False)
    dense_weight_file = torch.load('/home/zhouxiaoli/DDAI-TCNet/Weights/densenet121-a639ec97.pth')
    dense_base.load_state_dict(dense_weight_file, strict=False)
    
    vitamc_net = ViT-AMCNet_Ablation(vit_base=vit_base, dense_base=dense_base)
    print(summary(vitamc_net, torch.randn((1, 3, 224, 224))))
    print(vitamc_net)
    vitamc_net = vitamc_net.cuda(gpu_device)
    '''

    ###ViT-AMCNet experiments

    vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=class_num, embed_dim=768,
                                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    vit_weight_file = torch.load(r'E:\DDAI-TCNet-Github\Weights\jx_vit_base_p16_224-80ecf9dd.pth')
    vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    vit_base.load_state_dict(vit_weight_file, strict=False)

    dense_base = densenet121(pretrained=False)
    dense_weight_file = torch.load(r'E:\DDAI-TCNet-Github\Weights\densenet121-a639ec97.pth')
    dense_base.load_state_dict(dense_weight_file, strict=False)

    vitamc_net = ViT-AMCNet(vit_base=vit_base, dense_base=dense_base, work_mode=work_mode, model_mode=model_mode,
                          class_num = class_num)
    print(summary(vitamc_net, torch.randn((2, 3, 224, 224))))
    print(vitamc_net)
    vitamc_net = vitamc_net.cuda(gpu_device)


    ###Adding graphs or scalers to tensorboardX
    #write_1.add_graph(ddai_net, input_to_model=torch.ones(1, 3, 224, 224))
    #igcrga_net = IGCROI_GANet(3, base_net)


    ###Single model fitting section
    #Single_out_fit(ddai_net=vit_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    #               lr_fn='vit', epoch = 100, gpu_device = gpu_device)

    ###Double_out model fitting section
    #Double_out_fit(ddai_net=vitamc_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    #               epoch=100, gpu_device=gpu_device)

    ###Triplet_out model fitting section
    Triplet_out_fit(ddai_net=vitamc_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                  epoch=10, gpu_device=gpu_device, train_mode=train_mode)

    #save_model(model=vitamc_net, file_name='ViT-visual.pt', file_path='E:\DDAI-TCNet-Github\Weights\\Tian.pth')
    #model_weights = torch.load('E:\\DDAI-TCNet-Github\\Weights\\ablation_exp\\'+model_mode+'.pt',
    #                           map_location='cuda:0')
    #model_weights = torch.load(r'E:\DDAI-TCNet-Github\Weights\breast_exp\ViT.pth',
                               #map_location='cuda:0')
    #vitamc_net.load_state_dict(model_weights, strict=False)

    acc_scores(model=vitamc_net, gpu_device=gpu_device, data_loader=test_loader,
               out_mode='triplet', class_num = class_num)

    #print()

    ###Visually Interpretable for Rollout

    train_img = torch.zeros(2, 3, 224, 224)

    for i, j in train_loader:
        train_img = torch.cat((train_img, i), dim=0)

    train_img = train_img[2:]
    input_tensor = train_img[0].cuda(gpu_device)
    input_tensor = torch.reshape(input_tensor, [1, 3, 224, 224])
    attention_rollout = VITAttentionRollout(vitamc_net)
    heatmap = attention_rollout(input_tensor)

    np_img = torch.reshape(input_tensor, [3, 224, 224])
    np_img = torch.transpose(np_img, 0, 2).cpu().numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (np_img.shape[1], np_img.shape[0]))
    heatmap = np.uint8(255.0 * heatmap)
    # heatmap = np.abs(255.0 - heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap)

    superimposed_img = cv2.addWeighted(np_img, 0.9, heatmap, 0.1, 0)

    plt.figure(1)
    plt.imshow(superimposed_img)
    plt.savefig('E:\DDAI-TCNet-Github\\1.jpg')

    plt.figure(2)
    plt.imshow(np_img)
    plt.show()