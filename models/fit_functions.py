############################# ViT_AMCNet_main.py ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: A python script for running ViT_AMCNet

########################## API Section #########################

import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score
import numpy as np
from torch.optim import lr_scheduler

def cnn_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-4
    elif epoch < 75:
        lr = 2e-5
    else:
        lr = 1e-6
    return lr

def vit_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-5
    elif epoch < 75:
        lr = 5e-6
    else:
        lr = 1e-6
    return lr

def vit_lr_for_breast_schedule(epoch):
    if epoch < 50:
        lr = 2e-5
    elif epoch < 75:
        lr = 5e-6
    else:
        lr = 1e-6
    return lr

def fusion_lr_schedule(epoch):
    if epoch < 50:
        lr = 5e-5
    elif epoch < 75:
        lr = 2e-6
    else:
        lr = 1e-6
    return lr


def Single_out_fit(ddai_net=None, train_loader=None, val_loader=None, test_loader=None, lr_fn=None, epoch = 100,
                   gpu_device = 0):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=vit_lr_for_breast_schedule(i))

        ddai_net.train()
        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            pre_y = ddai_net(img_data)
            loss_value = loss_fn(pre_y, img_label)
            # loss_value = loss_dense + loss_vit
            loss_value.backward()
            # scheduler.step()
            rmp_optim.step()
            rmp_optim.zero_grad()

        ddai_net.eval()
        train_acc = []
        for train_img, train_label in train_loader:
            train_img = train_img.cuda(gpu_device)
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_pre_y = ddai_net(train_img)
                train_pre_label = torch.argmax(train_pre_y, dim=1)
                train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    train_pre_label.detach().cpu().numpy()))

        val_acc = []
        for val_img, val_label in val_loader:
            val_img = val_img.cuda(gpu_device)
            val_label = val_label.cuda(gpu_device)
            with torch.no_grad():
                val_pre_y = ddai_net(val_img)
                val_loss = loss_fn(val_pre_y, val_label)
                val_pre_label = torch.argmax(val_pre_y, dim=1)
                val_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  val_pre_label.detach().cpu().numpy()))

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(loss_value.detach().cpu().numpy()),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(val_loss.detach().cpu().numpy()),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    ddai_net.eval()
    test_acc = []
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            test_pre_y = ddai_net(test_img)
            test_loss = loss_fn(test_pre_y, test_label)
            test_pre_label = torch.argmax(test_pre_y, dim=1)
            test_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               test_pre_label.detach().cpu().numpy()))

    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))
    # write_1.add_graph(ddai_net, input_to_model=train_img[0:2])
    # g = ddai_net.state_dict()
    # torch.save(g, 'E:\DDAI-TCNet\Weights\Dense_SE.pth')


def Double_out_fit(ddai_net = None, train_loader = None, val_loader = None, test_loader = None, epoch = 100,
                   gpu_device = 0):
    ViT_loss = nn.CrossEntropyLoss()
    FABNet_loss = nn.CrossEntropyLoss()
    ddai_net = ddai_net.cuda(gpu_device)
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    for i in range(epoch):
        start_time = time.time()
        FABNet_para = [{'params': ddai_net.input_module.parameters()},
                       {'params': ddai_net.dense_block_1.parameters()}
            , {'params': ddai_net.dense_block_2.parameters()},
                       {'params': ddai_net.dense_block_3.parameters()}
            , {'params': ddai_net.dense_block_4.parameters()}, {'params': ddai_net.BN_5.parameters()}
            , {'params': ddai_net.transition_1.parameters()},
                       {'params': ddai_net.transition_2.parameters()}
            , {'params': ddai_net.transition_3.parameters()},
                       {'params': ddai_net.top_linear_3.parameters()}
            , {'params': ddai_net.SE_1.parameters()}, {'params': ddai_net.SE_2.parameters()}
            , {'params': ddai_net.SE_3.parameters()}, {'params': ddai_net.SE_4.parameters()}
            , {'params': ddai_net.NonLocal_1.parameters()}, {'params': ddai_net.NonLocal_2.parameters()}
            , {'params': ddai_net.NonLocal_3.parameters()}, {'params': ddai_net.NonLocal_4.parameters()}]

        ViT_para = [{'params': ddai_net.patch_embed.parameters()},
                    {'params': ddai_net.transformer_block_0.parameters()},
                    {'params': ddai_net.transformer_block_1.parameters()},
                    {'params': ddai_net.transformer_block_2.parameters()},
                    {'params': ddai_net.transformer_block_3.parameters()},
                    {'params': ddai_net.transformer_block_4.parameters()},
                    {'params': ddai_net.transformer_block_5.parameters()},
                    {'params': ddai_net.transformer_block_6.parameters()},
                    {'params': ddai_net.transformer_block_7.parameters()},
                    {'params': ddai_net.transformer_block_8.parameters()},
                    {'params': ddai_net.transformer_block_9.parameters()},
                    {'params': ddai_net.transformer_block_10.parameters()},
                    {'params': ddai_net.transformer_block_11.parameters()},
                    {'params': ddai_net.norm.parameters()},
                    {'params': ddai_net.head.parameters()}]

        rmp_vit_optim = torch.optim.RMSprop(ViT_para, lr=vit_lr_schedule(i))
        rmp_fab_optim = torch.optim.RMSprop(FABNet_para, lr=cnn_lr_schedule(i))
        ddai_net.train()
        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            vit_y, dense_y = ddai_net(img_data)
            loss_ViT = ViT_loss(vit_y, img_label)
            loss_FABNet = FABNet_loss(dense_y, img_label)
            # loss_value = loss_dense + loss_vit
            loss_ViT.backward()
            loss_FABNet.backward()
            # scheduler.step()
            rmp_vit_optim.step()
            rmp_vit_optim.zero_grad()
            rmp_fab_optim.step()
            rmp_fab_optim.zero_grad()

        ddai_net.eval()
        train_FAB_acc = []
        train_ViT_acc = []
        for train_img, train_label in train_loader:
            train_img = train_img.cuda(gpu_device)
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_ViT_y, train_FAB_y = ddai_net(train_img)
                pre_train_FAB = torch.argmax(train_FAB_y, dim=1)
                pre_train_ViT = torch.argmax(train_ViT_y, dim=1)
                train_FAB_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    pre_train_FAB.detach().cpu().numpy()))
                train_ViT_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    pre_train_ViT.detach().cpu().numpy()))

        val_FAB_acc = []
        val_ViT_acc = []
        for val_img, val_label in val_loader:
            val_img = val_img.cuda(gpu_device)
            val_label = val_label.cuda(gpu_device)
            with torch.no_grad():
                val_ViT_y, val_FAB_y = ddai_net(val_img)
                val_FAB_loss = FABNet_loss(val_FAB_y, val_label)
                val_ViT_loss = ViT_loss(val_ViT_y, val_label)
                pre_val_FAB = torch.argmax(val_FAB_y, dim=1)
                pre_val_ViT = torch.argmax(val_ViT_y, dim=1)
                val_FAB_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  pre_val_FAB.detach().cpu().numpy()))
                val_ViT_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  pre_val_ViT.detach().cpu().numpy()))

        end_time = time.time()
        print('epoch ' + str(i + 1), ' train_FAB_loss:{:.4}'.format(loss_FABNet.detach().cpu().numpy()),
              ' train_FAB_acc:{:.4}'.format(np.mean(train_FAB_acc)),
              ' val_FAB_loss:{:.4}'.format(val_FAB_loss.detach().cpu().numpy()),
              ' val_FAB_acc:{:.4}'.format(np.mean(val_FAB_acc)),
              ' Time:{:.3}'.format(end_time - start_time), '\n',
              '________train_ViT_loss:{:.4}'.format(loss_ViT.detach().cpu().numpy()),
              ' train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
              '   val_ViT_loss:{:.4}'.format(val_ViT_loss.detach().cpu().numpy()),
              ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    ddai_net.eval()
    test_FAB_acc = []
    test_ViT_acc = []
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            test_ViT_y, test_FAB_y = ddai_net(test_img)
            test_FAB_loss = FABNet_loss(test_FAB_y, test_label)
            pre_test_FAB = torch.argmax(test_FAB_y, dim=1)
            test_FAB_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               pre_test_FAB.detach().cpu().numpy()))

            test_ViT_loss = ViT_loss(test_ViT_y, test_label)
            pre_test_ViT = torch.argmax(test_ViT_y, dim=1)
            test_ViT_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               pre_test_ViT.detach().cpu().numpy()))

    print('train_FAB_acc:{:.4}'.format(np.mean(train_FAB_acc)),
          ' val_FAB_acc:{:.4}'.format(np.mean(val_FAB_acc)),
          ' test_FAB_acc:{:.4}'.format(np.mean(test_FAB_acc)), '\n',
          'train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
          ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
          ' test_ViT_acc:{:.4}'.format(np.mean(test_ViT_acc)))
    # write_1.add_graph(ddai_net, input_to_model=train_img[0:2])
    # g = ddai_net.state_dict()
    # torch.save(g, 'E:\DDAI-TCNet\Weights\Dense_SE.pth')

def Triplet_out_fit(ddai_net = None, train_loader = None, val_loader = None, test_loader = None, epoch = 100,
                   gpu_device = 0, train_mode = None):
    ViT_loss = nn.CrossEntropyLoss()
    FABNet_loss = nn.CrossEntropyLoss()
    Fusion_loss = nn.CrossEntropyLoss()
    Triplet_loss = nn.TripletMarginLoss()
    ddai_net = ddai_net.cuda(gpu_device)
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    for i in range(epoch):
        start_time = time.time()
        if train_mode == 'FDAF':
            para = [{'params': ddai_net.input_module.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.dense_block_1.parameters(), 'lr':cnn_lr_schedule(i)}
                , {'params': ddai_net.dense_block_2.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.dense_block_3.parameters(), 'lr':cnn_lr_schedule(i)}
                , {'params': ddai_net.dense_block_4.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.BN_5.parameters(), 'lr':cnn_lr_schedule(i)}
                , {'params': ddai_net.transition_1.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.transition_2.parameters(), 'lr':cnn_lr_schedule(i)}
                , {'params': ddai_net.transition_3.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.top_linear_3.parameters(), 'lr':cnn_lr_schedule(i)}
                , {'params': ddai_net.SE_1.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.SE_2.parameters(), 'lr':cnn_lr_schedule(i)}
                , {'params': ddai_net.SE_3.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.SE_4.parameters(), 'lr':cnn_lr_schedule(i)}
                , {'params': ddai_net.NonLocal_1.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.NonLocal_2.parameters(), 'lr':cnn_lr_schedule(i)}
                , {'params': ddai_net.NonLocal_3.parameters()},
                           {'params': ddai_net.NonLocal_4.parameters(), 'lr':cnn_lr_schedule(i)},
                           {'params': ddai_net.patch_embed.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_0.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_1.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_2.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_3.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_4.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_5.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_6.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_7.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_8.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_9.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_10.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.transformer_block_11.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.norm.parameters(), 'lr':vit_lr_schedule(i)},
                        {'params': ddai_net.head.parameters(), 'lr':vit_lr_schedule(i)}
                        ,{'params': ddai_net.fusion_layer.parameters(), 'lr':cnn_lr_schedule(i)}
                        ,{'params': ddai_net.fusion_block_dense_1.parameters(), 'lr':cnn_lr_schedule(i)}
                        ,{'params': ddai_net.fusion_block_dense_2.parameters(), 'lr':cnn_lr_schedule(i)}]
        elif train_mode == 'FDAI' or train_mode == 'MLMT':
            para = [{'params': ddai_net.input_module.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.dense_block_1.parameters(), 'lr': cnn_lr_schedule(i)}
                , {'params': ddai_net.dense_block_2.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.dense_block_3.parameters(), 'lr': cnn_lr_schedule(i)}
                , {'params': ddai_net.dense_block_4.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.BN_5.parameters(), 'lr': cnn_lr_schedule(i)}
                , {'params': ddai_net.transition_1.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.transition_2.parameters(), 'lr': cnn_lr_schedule(i)}
                , {'params': ddai_net.transition_3.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.top_linear_3.parameters(), 'lr': cnn_lr_schedule(i)}
                , {'params': ddai_net.SE_1.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.SE_2.parameters(), 'lr': cnn_lr_schedule(i)}
                , {'params': ddai_net.SE_3.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.SE_4.parameters(), 'lr': cnn_lr_schedule(i)}
                , {'params': ddai_net.NonLocal_1.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.NonLocal_2.parameters(), 'lr': cnn_lr_schedule(i)}
                , {'params': ddai_net.NonLocal_3.parameters()},
                         {'params': ddai_net.NonLocal_4.parameters(), 'lr': cnn_lr_schedule(i)},
                         {'params': ddai_net.patch_embed.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
                         {'params': ddai_net.transformer_block_0.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_1.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_3.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_4.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_5.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_6.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_7.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_8.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_9.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_10.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.transformer_block_11.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.norm.parameters(), 'lr': vit_lr_schedule(i)},
                         {'params': ddai_net.head.parameters(), 'lr': vit_lr_schedule(i)}
                , {'params': ddai_net.fusion_layer.parameters(), 'lr': cnn_lr_schedule(i)}]
        elif train_mode == 'Featrues_stacking' or train_mode =='Probability_fusion':
            para = [{'params': ddai_net.parameters(), 'lr': cnn_lr_schedule(i)}]

        rmp_optim = torch.optim.RMSprop(para)

        ddai_net.train()
        print('########################## training model #########################')
        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            vit_y, dense_y, fusion_y = ddai_net(img_data)
            loss_ViT = ViT_loss(vit_y, img_label)
            loss_FABNet = FABNet_loss(dense_y, img_label)
            loss_Fusion = Fusion_loss(fusion_y, img_label)
            loss_Triplet = Triplet_loss(fusion_y, dense_y, vit_y)

            if train_mode == 'MLMT':
                ###Multi-Learning rate learning and Multi-Target optimization
                ###注意，这里的损失函数的权重取值与Torch版本和cuda版本有关系，例如Torch1.6（0.5,0.5,1.0）和Torch1.9（0.5,1.0,0.5）
                loss_sum =  0.5 * loss_Fusion + 0.5 * loss_ViT + loss_FABNet
            elif train_mode == 'FDAI':
                ####Multi-Learning rate learning and Multi-Target optimization + Probability Distribution Metric Learning
                ###注意，上述损失函数权重取值原则仍适用，不同Torch和CUDA版本，请调参
                loss_sum =  0.5 * loss_Fusion + 0.5 * loss_ViT + loss_FABNet + 0.2 * loss_Triplet
            elif train_mode == 'FDAF':
                ####Multi-Learning rate learning and Multi-Target optimization + Probability Distribution Metric Learning +
                ###Adaptive Features Fusion
                ###注意，上述损失函数权重取值原则仍适用，不同Torch和CUDA版本，请调参
                loss_sum = 0.2 * loss_Fusion + 0.5 * loss_ViT + loss_FABNet + 0.2 * loss_Triplet


            if train_mode == 'FDAI' or train_mode == 'MLMT' or train_mode == 'FDAF':
                loss_sum.backward()
            elif train_mode == 'Probability_fusion' or train_mode == 'Featrues_stacking':
                loss_Fusion.backward()

            rmp_optim.step()
            rmp_optim.zero_grad()


        ddai_net.eval()
        train_FAB_acc = []
        train_ViT_acc = []
        train_Fusion_acc = []
        #train_Sum_acc = []
        for train_img, train_label in train_loader:
            train_img = train_img.cuda(gpu_device)
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_ViT_y, train_FAB_y, train_Fusion_y = ddai_net(train_img)
                pre_train_FAB = torch.argmax(train_FAB_y, dim=1)
                pre_train_ViT = torch.argmax(train_ViT_y, dim=1)
                pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
                #pre_train_Sum = torch.argmax(train_Sum_y , dim=1)
                train_FAB_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    pre_train_FAB.detach().cpu().numpy()))
                train_ViT_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    pre_train_ViT.detach().cpu().numpy()))
                train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    pre_train_Fusion.detach().cpu().numpy()))
                #train_Sum_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                #                                       pre_train_Sum.detach().cpu().numpy()))

        val_FAB_acc = []
        val_ViT_acc = []
        val_Fusion_acc = []
        #val_Sum_acc = []
        for val_img, val_label in val_loader:
            val_img = val_img.cuda(gpu_device)
            val_label = val_label.cuda(gpu_device)
            with torch.no_grad():
                val_ViT_y, val_FAB_y, val_Fusion_y = ddai_net(val_img)
                val_FAB_loss = FABNet_loss(val_FAB_y, val_label)
                val_ViT_loss = ViT_loss(val_ViT_y, val_label)
                val_Fusion_loss = ViT_loss(val_Fusion_y, val_label)
                pre_val_FAB = torch.argmax(val_FAB_y, dim=1)
                pre_val_ViT = torch.argmax(val_ViT_y, dim=1)
                pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
                #pre_val_Sum = torch.argmax(val_Sum_y, dim=1)
                val_FAB_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  pre_val_FAB.detach().cpu().numpy()))
                val_ViT_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  pre_val_ViT.detach().cpu().numpy()))
                val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  pre_val_Fusion.detach().cpu().numpy()))
                #val_Sum_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                #                                     pre_val_Sum.detach().cpu().numpy()))

        end_time = time.time()
        print('epoch ' + str(i + 1), ' train_FAB_loss:{:.4}'.format(loss_FABNet.detach().cpu().numpy()),
              ' train_FAB_acc:{:.4}'.format(np.mean(train_FAB_acc)),
              ' val_FAB_loss:{:.4}'.format(val_FAB_loss.detach().cpu().numpy()),
              ' val_FAB_acc:{:.4}'.format(np.mean(val_FAB_acc)),
              ' Time:{:.3}'.format(end_time - start_time), '\n',
              '________train_ViT_loss:{:.4}'.format(loss_ViT.detach().cpu().numpy()),
              ' train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
              '   val_ViT_loss:{:.4}'.format(val_ViT_loss.detach().cpu().numpy()),
              ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)), '\n',
            '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
            ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
            '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
            ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    ddai_net.eval()
    test_FAB_acc = []
    test_ViT_acc = []
    test_Fusion_acc = []
    #test_Sum_acc = []
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            test_ViT_y, test_FAB_y, test_Fusion_y = ddai_net(test_img)
            test_FAB_loss = FABNet_loss(test_FAB_y, test_label)
            pre_test_FAB = torch.argmax(test_FAB_y, dim=1)
            test_FAB_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               pre_test_FAB.detach().cpu().numpy()))

            test_ViT_loss = ViT_loss(test_ViT_y, test_label)
            pre_test_ViT = torch.argmax(test_ViT_y, dim=1)
            test_ViT_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               pre_test_ViT.detach().cpu().numpy()))

            test_Fusion_loss = Fusion_loss(test_Fusion_y, test_label)
            pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
            test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               pre_test_Fusion.detach().cpu().numpy()))

            #pre_test_Sum = torch.argmax(test_Sum_y, dim=1)
            #test_Sum_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
            #                                      pre_test_Sum.detach().cpu().numpy()))
    print('\n', '########################## testing model #########################')
    print('train_FAB_acc:{:.4}'.format(np.mean(train_FAB_acc)),
          ' val_FAB_acc:{:.4}'.format(np.mean(val_FAB_acc)),
          ' test_FAB_acc:{:.4}'.format(np.mean(test_FAB_acc)), '\n',
          'train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
          ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
          ' test_ViT_acc:{:.4}'.format(np.mean(test_ViT_acc)), '\n',
          'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
          ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
          ' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
    # write_1.add_graph(ddai_net, input_to_model=train_img[0:2])
    # g = ddai_net.state_dict()
    # torch.save(g, 'E:\DDAI-TCNet\Weights\Dense_SE.pth')