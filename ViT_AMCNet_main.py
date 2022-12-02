############################# ViT_AMCNet_main.py ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: A python script for running ViT_AMCNet

########################## API Section #########################
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
from models.ViT import VisionTransformer, load_pretrained, vit_base_patch16_224, vit_small_patch16_224,\
    vit_base_patch32_224, vit_large_patch16_224, vit_large_patch32_224
from models.model_modules import ViT_AMCNet_Ablation, Dense_Net, FABNet, ViT_Net, ViT_AMCNet, \
    ViT_Large_Net, ViT_Small_Net, Softmax
from models.fit_functions import Double_out_fit, vit_lr_schedule, Single_out_fit, Triplet_out_fit
from models.ablation_experiments import save_model, acc_scores, to_np_category
from models.vit_grad_rollout import VITAttentionGradRollout
from models.vit_rollout import VITAttentionRollout, show_mask_on_image
import cv2
from skimage import io
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

########################## seed_function #########################
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
    ########################## Hyparameters #########################
    setup_seed(1234)
    gpu_device = 0
    class_num = 3
    work_mode = 'normal'  # work_mode = {'normal', 'features_extraction'}
    train_mode = 'FDAF'   # train_mode = {'MLMT', 'FDAI', 'FDAF', 'Features_stacking', 'Probability_fusion'}}
    model_mode = 'FDAF'   # model_mode = {'MLMT', 'FDAI', 'FDAF', 'Features_stacking', 'Probability_fusion'}
    epochs = 1
    batch_size = 16
    image_size = [224, 224]

    ########################## reading datas and processing datas #########################
    print('########################## reading datas and processing datas #########################')
    transform = transforms.Compose([transforms.Resize(image_size) ,transforms.ToTensor()
                                   , transforms.Normalize(mean=0.5, std=0.5)])
    train_data = ImageFolder(r'F:\ViT-AMCNet_pytorch\Datasets\Larynx_dataset\Train_patch',
                             transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_data = ImageFolder(r'F:\ViT-AMCNet_pytorch\Datasets\Larynx_dataset\Test_patch',
                            transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)
    val_data = ImageFolder(r'F:\ViT-AMCNet_pytorch\Datasets\Larynx_dataset\Val_patch',
                            transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1)
    print(train_data)

    ########################## creating a FABNet model #########################
    #print('########################## creating a FABNet model #########################')
    #dense_base = densenet121(pretrained=False)
    #dense_weight_file = torch.load(r'E:\ViT-AMCNet_pytorch\Weights\densenet121-a639ec97.pth')
    #dense_base.load_state_dict(dense_weight_file, strict=False)
    #fab_net = FABNet(base_net=dense_base, class_num=class_num)
    #print(summary(fab_net, torch.randn((1, 3, 224, 224))))
    #print(fab_net)
    #fab_net = fab_net.cuda(gpu_device)

    ########################## creating a general ViT model #########################
    #print('########################## creating a general ViT model #########################')
    #vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=class_num, embed_dim=768,
    #            depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
    #            attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    #vit_weight_file = torch.load(r'E:\ViT-AMCNet_pytorch\Weights\jx_vit_base_p16_224-80ecf9dd.pth')
    #vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    #vit_base.load_state_dict(vit_weight_file, strict=False)
    #vit_net = ViT_Net(base_net=vit_base, class_num=class_num)
    #print(summary(vit_net , torch.randn((1, 3, 224, 224))))
    #print(vit_net)
    #vit_net  = vit_net.cuda(gpu_device)

    ########################## creating a ViT_AMCNet_Ablation model #########################
    #print('########################## creating a ViT_AMCNet_Ablation model #########################')
    #vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=3, embed_dim=768,
    #                             depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
    #                             attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    #vit_weight_file = torch.load(r'F:\ViT-AMCNet_pytorch\Weights\jx_vit_base_p16_224-80ecf9dd.pth')
    #vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    #vit_base.load_state_dict(vit_weight_file, strict=False)
    #dense_base = densenet121(pretrained=False)
    #dense_weight_file = torch.load(r'F:\ViT-AMCNet_pytorch\Weights\densenet121-a639ec97.pth')
    #dense_base.load_state_dict(dense_weight_file, strict=False)
    #vit_amcnet = ViT_AMCNet_Ablation(vit_base=vit_base, dense_base=dense_base)
    #print(summary(vit_amcnet, torch.randn((1, 3, 224, 224))))
    #print(vit_amcnet)
    #vit_amcnet = vit_amcnet.cuda(gpu_device)

    ########################## creating a ViT_AMCNet model #########################
    print('########################## creating a ViT_AMCNet model #########################')
    vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=class_num, embed_dim=768,
                                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    vit_weight_file = torch.load(r'F:\ViT-AMCNet_pytorch\Weights\jx_vit_base_p16_224-80ecf9dd.pth')
    vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    vit_base.load_state_dict(vit_weight_file, strict=False)

    dense_base = densenet121(pretrained=False)
    dense_weight_file = torch.load(r'F:\ViT-AMCNet_pytorch\Weights\densenet121-a639ec97.pth')
    dense_base.load_state_dict(dense_weight_file, strict=False)

    vit_amcnet = ViT_AMCNet(vit_base=vit_base, dense_base=dense_base, work_mode=work_mode, model_mode=model_mode,
                          class_num = class_num)
    with torch.no_grad():
        print('########################## ViT-AMCNet_summary #########################')
        print(summary(vit_amcnet, torch.randn((2, 3, 224, 224))))
        print('\n', '########################## ViT-AMCNet #########################')
        print(vit_amcnet)
    vit_amcnet = vit_amcnet.cuda(gpu_device)

    ########################## #Adding graphs or scalers to tensorboardX #########################
    #write_1 = SummaryWriter('/home/zhouxiaoli/DDAI-TCNet/Log/Dense_SE')
    #write_1.add_graph(vit_amcnet, input_to_model=torch.ones(1, 3, 224, 224))

    ########################## fitting models and testing models #########################
    print('########################## fitting models and testing models #########################')
    ########################## Single_out fitting #########################
    #Single_out_fit(ddai_net=fab_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    #               lr_fn='cnn', epoch = 100, gpu_device = gpu_device)

    ########################## Double_out fitting #########################
    #Double_out_fit(ddai_net=ddai_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    #               epoch=100, gpu_device=gpu_device)

    ########################## Triplet_out fitting #########################
    Triplet_out_fit(ddai_net=vit_amcnet, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                  epoch=epochs, gpu_device=gpu_device, train_mode=train_mode)

    ########################### Drawing ROC Curves #########################
    #test_pre_y, test_label = acc_scores(model=vit_amcnet, gpu_device=gpu_device, data_loader=test_loader,
    #           out_mode='triplet', class_num = class_num)

    #print(test_pre_y.shape)
    #print(test_label.shape)
    #softmax_model = Softmax()

    #test_pre_y = softmax_model(test_pre_y)

    #test_label = np.array(test_label.cpu().detach())
    #test_pre_y = np.array(test_pre_y.cpu().detach())
    #test_pre_label = np.argmax(test_pre_y,axis=1)
    #print(accuracy_score(test_label, test_pre_label))
    #roc_true = to_np_category(test_label).ravel()
    #roc_pre = to_np_category(test_pre_label).ravel()
    #fpr, tpr, _ = roc_curve(roc_true, roc_pre)
    #plt.figure(1)
    #plt.plot(fpr, tpr)
    #plt.show()
    #np.save('E:\\fpr_model.npy', fpr)
    #np.save('E:\\tpr_model.npy', tpr)

    ########################### Visually Interpretable for Rollout #########################
    #train_img = torch.zeros(2, 3, 224, 224)

    #for i, j in test_loader:
    #    train_img = torch.cat((train_img, i), dim=0)

    #train_img = train_img[2:]
    #input_tensor = train_img[156].cuda(gpu_device)
    #input_tensor = torch.reshape(input_tensor, [1, 3, 224, 224])
    #attention_rollout = VITAttentionRollout(vit_amcnet)
    #heatmap = attention_rollout(input_tensor)
    #np_img = torch.reshape(input_tensor, [3, 224, 224])
    #np_img = np_img.cpu().numpy()
    #np_img = np.transpose(np_img, (1, 2, 0))
    #heatmap = np.maximum(heatmap, 0)
    #heatmap /= np.max(heatmap)
    #heatmap = cv2.resize(heatmap, (np_img.shape[0], np_img.shape[0]))
    #heatmap = np.uint8(255.0 * heatmap)
    #heatmap = np.abs(255.0 - heatmap)
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #heatmap = np.float32(heatmap)
    #superimposed_img = cv2.addWeighted(np_img, 0.9, heatmap, 0.1, 0)

    #plt.figure(2)
    #plt.imshow(superimposed_img)
    #plt.savefig('E:\DDAI-TCNet-Github\\1.jpg')

    #plt.figure(3)
    #plt.imshow(np_img)
    #plt.show()














