############################# model_modules.py ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: A python script for creating various models

import torch
from torch import nn
from models.other_modules import NONLocalBlock2D, SELayer

class Dense_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(Dense_Net, self).__init__()
        self.base_net = base_net.features
        self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
        self.top_flat = nn.Flatten()
        self.top_linear_3 = nn.Linear(1024, class_num)
        #nn.init.kaiming_normal(self.top_linear_3.weight)

    def forward(self, x):
        y = self.base_net(x)
        y = self.top_AvgMp(y)
        y = self.top_flat(y)
        y = self.top_linear_3(y)
        return y

class ViT_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(ViT_Net, self).__init__()
        self.patch_embed = base_net.patch_embed
        self.cls_token = base_net.cls_token
        self.pos_embed = base_net.pos_embed
        self.pos_drop = base_net.pos_drop
        self.transformer_block_0 = base_net.blocks[0]
        self.transformer_block_1 = base_net.blocks[1]
        self.transformer_block_2 = base_net.blocks[2]
        self.transformer_block_3 = base_net.blocks[3]
        self.transformer_block_4 = base_net.blocks[4]
        self.transformer_block_5 = base_net.blocks[5]
        self.transformer_block_6 = base_net.blocks[6]
        self.transformer_block_7 = base_net.blocks[7]
        self.transformer_block_8 = base_net.blocks[8]
        self.transformer_block_9 = base_net.blocks[9]
        self.transformer_block_10 = base_net.blocks[10]
        self.transformer_block_11 = base_net.blocks[11]
        self.norm = base_net.norm
        self.head = base_net.head

    def forward(self, x):
        y = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        y  = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.pos_drop(y)
        y = self.transformer_block_0(y)
        y = self.transformer_block_1(y)
        y = self.transformer_block_2(y)
        y = self.transformer_block_3(y)
        y = self.transformer_block_4(y)
        y = self.transformer_block_5(y)
        y = self.transformer_block_6(y)
        y = self.transformer_block_7(y)
        y = self.transformer_block_8(y)
        y = self.transformer_block_9(y)
        y = self.transformer_block_10(y)
        y = self.transformer_block_11(y)
        y = self.norm(y)
        y = self.head(y)
        return y[:, 0, :]

class ViT_Small_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(ViT_Small_Net, self).__init__()
        self.patch_embed = base_net.patch_embed
        self.cls_token = base_net.cls_token
        self.pos_embed = base_net.pos_embed
        self.pos_drop = base_net.pos_drop
        self.transformer_block_0 = base_net.blocks[0]
        self.transformer_block_1 = base_net.blocks[1]
        self.transformer_block_2 = base_net.blocks[2]
        self.transformer_block_3 = base_net.blocks[3]
        self.transformer_block_4 = base_net.blocks[4]
        self.transformer_block_5 = base_net.blocks[5]
        self.transformer_block_6 = base_net.blocks[6]
        self.transformer_block_7 = base_net.blocks[7]
        self.norm = base_net.norm
        self.head = base_net.head

    def forward(self, x):
        y = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        y  = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.pos_drop(y)
        y = self.transformer_block_0(y)
        y = self.transformer_block_1(y)
        y = self.transformer_block_2(y)
        y = self.transformer_block_3(y)
        y = self.transformer_block_4(y)
        y = self.transformer_block_5(y)
        y = self.transformer_block_6(y)
        y = self.transformer_block_7(y)
        y = self.norm(y)
        y = self.head(y)
        return y[:, 0, :]


class ViT_Large_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(ViT_Large_Net, self).__init__()
        self.patch_embed = base_net.patch_embed
        self.cls_token = base_net.cls_token
        self.pos_embed = base_net.pos_embed
        self.pos_drop = base_net.pos_drop
        self.transformer_block_0 = base_net.blocks[0]
        self.transformer_block_1 = base_net.blocks[1]
        self.transformer_block_2 = base_net.blocks[2]
        self.transformer_block_3 = base_net.blocks[3]
        self.transformer_block_4 = base_net.blocks[4]
        self.transformer_block_5 = base_net.blocks[5]
        self.transformer_block_6 = base_net.blocks[6]
        self.transformer_block_7 = base_net.blocks[7]
        self.transformer_block_8 = base_net.blocks[8]
        self.transformer_block_9 = base_net.blocks[9]
        self.transformer_block_10 = base_net.blocks[10]
        self.transformer_block_11 = base_net.blocks[11]
        self.transformer_block_12 = base_net.blocks[12]
        self.transformer_block_13 = base_net.blocks[13]
        self.transformer_block_14 = base_net.blocks[14]
        self.transformer_block_15 = base_net.blocks[15]
        self.transformer_block_16 = base_net.blocks[16]
        self.transformer_block_17 = base_net.blocks[17]
        self.transformer_block_18 = base_net.blocks[18]
        self.transformer_block_19 = base_net.blocks[19]
        self.transformer_block_20 = base_net.blocks[20]
        self.transformer_block_21 = base_net.blocks[21]
        self.transformer_block_22 = base_net.blocks[22]
        self.transformer_block_23 = base_net.blocks[23]
        self.norm = base_net.norm
        self.head = base_net.head

    def forward(self, x):
        y = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        y  = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.pos_drop(y)
        y = self.transformer_block_0(y)
        y = self.transformer_block_1(y)
        y = self.transformer_block_2(y)
        y = self.transformer_block_3(y)
        y = self.transformer_block_4(y)
        y = self.transformer_block_5(y)
        y = self.transformer_block_6(y)
        y = self.transformer_block_7(y)
        y = self.transformer_block_8(y)
        y = self.transformer_block_9(y)
        y = self.transformer_block_10(y)
        y = self.transformer_block_11(y)
        y = self.transformer_block_12(y)
        y = self.transformer_block_13(y)
        y = self.transformer_block_14(y)
        y = self.transformer_block_15(y)
        y = self.transformer_block_16(y)
        y = self.transformer_block_17(y)
        y = self.transformer_block_18(y)
        y = self.transformer_block_19(y)
        y = self.transformer_block_20(y)
        y = self.transformer_block_21(y)
        y = self.transformer_block_22(y)
        y = self.transformer_block_23(y)
        y = self.norm(y)
        y = self.head(y)
        return y[:, 0, :]


class FABNet(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(FABNet, self).__init__()
        #self.base_net = base_net.features
        self.input_module = base_net.features[0:4]
        self.dense_block_1 = base_net.features.denseblock1
        self.dense_block_2 = base_net.features.denseblock2
        self.dense_block_3 = base_net.features.denseblock3
        self.dense_block_4 = base_net.features.denseblock4
        self.transition_1 = base_net.features.transition1
        self.transition_2 = base_net.features.transition2
        self.transition_3 = base_net.features.transition3
        self.BN_5 = base_net.features.norm5

        self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
        self.top_flat = nn.Flatten()
        #self.top_linear_1 = nn.Linear(1024 * 7 * 7, 1024)
        self.top_linear_3 = nn.Linear(1024, class_num)
        self.dp = nn.Dropout(p=0.3)
        self.SE_1 = SELayer(128, 8)
        self.SE_2 = SELayer(256, 16)
        self.SE_3 = SELayer(512, 24)
        self.SE_4 = SELayer(1024, 32)
        self.NonLocal_1 = NONLocalBlock2D(in_channels=128)
        self.NonLocal_2 = NONLocalBlock2D(in_channels=256)
        self.NonLocal_3 = NONLocalBlock2D(in_channels=512)
        self.NonLocal_4 = NONLocalBlock2D(in_channels=1024)
        #nn.init.kaiming_normal(self.top_linear_3.weight)

    def forward(self, x):
        #y = self.base_net(x)
        y = self.input_module(x)
        y = self.dense_block_1(y)
        y = self.transition_1(y)
        y_1 = self.SE_1(y)
        y_2 = self.NonLocal_1(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_2(y)
        y = self.transition_2(y)
        y_1 = self.SE_2(y)
        y_2 = self.NonLocal_2(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_3(y)
        y = self.transition_3(y)
        y_1 = self.SE_3(y)
        y_2 = self.NonLocal_3(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_4(y)
        y = self.BN_5(y)
        y_1 = self.SE_4(y)
        y_2 = self.NonLocal_4(y)
        y = (y_1 + y_2) / 2.0
        y = self.top_AvgMp(y)
        y = self.top_flat(y)
        #y = self.dp(y)
        #y = self.top_linear_1(y)
        #y = self.top_linear_2(y)
        y = self.top_linear_3(y)
        return y

class ViT_AMCNet_Ablation(nn.Module):
    def __init__(self, vit_base = None, dense_base = None):
        super(ViT_AMCNet_Ablation, self).__init__()
        self.patch_embed = vit_base.patch_embed
        self.cls_token = vit_base.cls_token
        self.pos_embed = vit_base.pos_embed
        self.pos_drop = vit_base.pos_drop
        self.transformer_block_0 = vit_base.blocks[0]
        self.transformer_block_1 = vit_base.blocks[1]
        self.transformer_block_2 = vit_base.blocks[2]
        self.transformer_block_3 = vit_base.blocks[3]
        self.transformer_block_4 = vit_base.blocks[4]
        self.transformer_block_5 = vit_base.blocks[5]
        self.transformer_block_6 = vit_base.blocks[6]
        self.transformer_block_7 = vit_base.blocks[7]
        self.transformer_block_8 = vit_base.blocks[8]
        self.transformer_block_9 = vit_base.blocks[9]
        self.transformer_block_10 = vit_base.blocks[10]
        self.transformer_block_11 = vit_base.blocks[11]
        self.norm = vit_base.norm
        self.head = vit_base.head

        self.input_module = dense_base.features[0:4]
        self.dense_block_1 = dense_base.features.denseblock1
        self.dense_block_2 = dense_base.features.denseblock2
        self.dense_block_3 = dense_base.features.denseblock3
        self.dense_block_4 = dense_base.features.denseblock4
        self.transition_1 = dense_base.features.transition1
        self.transition_2 = dense_base.features.transition2
        self.transition_3 = dense_base.features.transition3
        self.BN_5 = dense_base.features.norm5
        self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
        self.top_flat = nn.Flatten()
        self.top_linear_3 = nn.Linear(1024, 3)
        self.dp = nn.Dropout(p=0.3)
        self.SE_1 = SELayer(128, 8)
        self.SE_2 = SELayer(256, 16)
        self.SE_3 = SELayer(512, 24)
        self.SE_4 = SELayer(1024, 32)
        self.NonLocal_1 = NONLocalBlock2D(in_channels=128)
        self.NonLocal_2 = NONLocalBlock2D(in_channels=256)
        self.NonLocal_3 = NONLocalBlock2D(in_channels=512)
        self.NonLocal_4 = NONLocalBlock2D(in_channels=1024)


    def forward(self, x):
        vit_y = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        vit_y = torch.cat((cls_tokens, vit_y), dim=1)
        vit_y = vit_y + self.pos_embed
        vit_y = self.pos_drop(vit_y)
        vit_y = self.transformer_block_0(vit_y)
        vit_y = self.transformer_block_1(vit_y)
        vit_y = self.transformer_block_2(vit_y)
        vit_y = self.transformer_block_3(vit_y)
        vit_y = self.transformer_block_4(vit_y)
        vit_y = self.transformer_block_5(vit_y)
        vit_y = self.transformer_block_6(vit_y)
        vit_y = self.transformer_block_7(vit_y)
        vit_y = self.transformer_block_8(vit_y)
        vit_y = self.transformer_block_9(vit_y)
        vit_y = self.transformer_block_10(vit_y)
        vit_y = self.transformer_block_11(vit_y)
        vit_y = self.norm(vit_y)
        vit_y = self.head(vit_y)
        vit_y = vit_y[:, 0, :]

        y = self.input_module(x)
        y = self.dense_block_1(y)
        y = self.transition_1(y)
        y_1 = self.SE_1(y)
        y_2 = self.NonLocal_1(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_2(y)
        y = self.transition_2(y)
        y_1 = self.SE_2(y)
        y_2 = self.NonLocal_2(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_3(y)
        y = self.transition_3(y)
        y_1 = self.SE_3(y)
        y_2 = self.NonLocal_3(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_4(y)
        y = self.BN_5(y)
        y_1 = self.SE_4(y)
        y_2 = self.NonLocal_4(y)
        y = (y_1 + y_2) / 2.0
        y = self.top_AvgMp(y)
        y = self.top_flat(y)
        # y = self.dp(y)
        # y = self.top_linear_1(y)
        # y = self.top_linear_2(y)
        dense_y = self.top_linear_3(y)
        return vit_y, dense_y

class ViT_AMCNet(nn.Module):
    def __init__(self, vit_base = None, dense_base = None, work_mode = None, model_mode = None, class_num = 3):
        super(ViT_AMCNet, self).__init__()
        self.patch_embed = vit_base.patch_embed
        self.cls_token = vit_base.cls_token
        self.pos_embed = vit_base.pos_embed
        self.pos_drop = vit_base.pos_drop
        self.transformer_block_0 = vit_base.blocks[0]
        self.transformer_block_1 = vit_base.blocks[1]
        self.transformer_block_2 = vit_base.blocks[2]
        self.transformer_block_3 = vit_base.blocks[3]
        self.transformer_block_4 = vit_base.blocks[4]
        self.transformer_block_5 = vit_base.blocks[5]
        self.transformer_block_6 = vit_base.blocks[6]
        self.transformer_block_7 = vit_base.blocks[7]
        self.transformer_block_8 = vit_base.blocks[8]
        self.transformer_block_9 = vit_base.blocks[9]
        self.transformer_block_10 = vit_base.blocks[10]
        self.transformer_block_11 = vit_base.blocks[11]
        self.norm = vit_base.norm
        self.head = vit_base.head

        self.input_module = dense_base.features[0:4]
        self.dense_block_1 = dense_base.features.denseblock1
        self.dense_block_2 = dense_base.features.denseblock2
        self.dense_block_3 = dense_base.features.denseblock3
        self.dense_block_4 = dense_base.features.denseblock4
        self.transition_1 = dense_base.features.transition1
        self.transition_2 = dense_base.features.transition2
        self.transition_3 = dense_base.features.transition3
        self.BN_5 = dense_base.features.norm5
        self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
        self.top_flat = nn.Flatten()
        self.top_linear_3 = nn.Linear(1024, class_num)
        self.dp = nn.Dropout(p=0.3)
        self.SE_1 = SELayer(128, 8)
        self.SE_2 = SELayer(256, 16)
        self.SE_3 = SELayer(512, 24)
        self.SE_4 = SELayer(1024, 32)
        self.NonLocal_1 = NONLocalBlock2D(in_channels=128)
        self.NonLocal_2 = NONLocalBlock2D(in_channels=256)
        self.NonLocal_3 = NONLocalBlock2D(in_channels=512)
        self.NonLocal_4 = NONLocalBlock2D(in_channels=1024)
        self.work_mode = work_mode
        self.model_mode = model_mode

        if self.model_mode == 'FDAF':
            self.fusion_block_dense_1 = nn.Linear(1792, 112)
            self.fusion_block_dense_2 = nn.Linear(112, 1792)
            self.fusion_layer = nn.Linear(1792, class_num)
        elif self.model_mode == 'FDAI' or self.model_mode =='MLMT' or self.model_mode =='Featrues_stacking':
            self.fusion_layer = nn.Linear(1792, class_num)

    def forward(self, x):
        vit_y = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        vit_y = torch.cat((cls_tokens, vit_y), dim=1)
        vit_y = vit_y + self.pos_embed
        vit_y = self.pos_drop(vit_y)
        vit_y = self.transformer_block_0(vit_y)
        vit_y = self.transformer_block_1(vit_y)
        vit_y = self.transformer_block_2(vit_y)
        vit_y = self.transformer_block_3(vit_y)
        vit_y = self.transformer_block_4(vit_y)
        vit_y = self.transformer_block_5(vit_y)
        vit_y = self.transformer_block_6(vit_y)
        vit_y = self.transformer_block_7(vit_y)
        vit_y = self.transformer_block_8(vit_y)
        vit_y = self.transformer_block_9(vit_y)
        vit_y = self.transformer_block_10(vit_y)
        vit_y = self.transformer_block_11(vit_y)
        vit_y_Fusion = self.norm(vit_y)
        vit_y = self.head(vit_y_Fusion)
        vit_y = vit_y[:, 0, :]

        y = self.input_module(x)
        y = self.dense_block_1(y)
        y = self.transition_1(y)
        y_1 = self.SE_1(y)
        y_2 = self.NonLocal_1(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_2(y)
        y = self.transition_2(y)
        y_1 = self.SE_2(y)
        y_2 = self.NonLocal_2(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_3(y)
        y = self.transition_3(y)
        y_1 = self.SE_3(y)
        y_2 = self.NonLocal_3(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_4(y)
        y = self.BN_5(y)
        y_1 = self.SE_4(y)
        y_2 = self.NonLocal_4(y)
        y = (y_1 + y_2) / 2.0
        y = self.top_AvgMp(y)
        y_Fusion = self.top_flat(y)
        fab_y = self.top_linear_3(y_Fusion)

        if self.model_mode == 'FDAF':
            ###Adaptive Features Fusion
            fusion_out = torch.cat((y_Fusion, vit_y_Fusion[:, 0, :]), dim=1)
            fusion_weights = self.fusion_block_dense_1(fusion_out)
            fusion_weights = torch.nn.functional.relu(fusion_weights)
            fusion_weights = self.fusion_block_dense_2(fusion_weights)
            fusion_weights = torch.nn.functional.softmax(fusion_weights, dim=1)
            fusion_out = fusion_out * fusion_weights
            fusion_out = self.fusion_layer(fusion_out)
        elif self.model_mode == 'FDAI' or self.model_mode == 'MLMT' or self.model_mode == 'Features_stacking':
            ###Features Fusion
            fusion_out = torch.cat((y_Fusion, vit_y_Fusion[:, 0, :]), dim=1)
            fusion_out = self.fusion_layer(fusion_out)
        elif self.model_mode == 'Probability_fusion':
            ###Probability Fusion
            fusion_out = fab_y + vit_y


        if self.work_mode == 'features_extraction':
            return y_Fusion, vit_y_Fusion[:, 0, :], fusion_out
        elif self.work_mode == 'normal':
            return vit_y, fab_y, fusion_out

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        y = self.softmax(x)
        return y


