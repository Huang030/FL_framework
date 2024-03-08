import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
import yaml
import logging
from utils import Helper

def local_train(local_model, global_model, dataloaders, epoch, param):
    # 所有客户端同时做local training，然后将权重求和返回
    weight_accumulator = dict()
    for key, value in global_model.state_dict().items():
        weight_accumulator[key] = torch.zeros_like(value)

    optimizer = torch.optim.Adam(local_model.parameters(), lr = param['lr'])
    loss_func = nn.CrossEntropyLoss()

    train_list = random.sample(range(param['parties']), param['num_train_parties'])
    print ()
    logger.info(f"the parties which are chosen: {train_list}")
    for i in range(param['parties']):
        if i in train_list:
            local_model.copy_params(global_model.state_dict())
            local_model.train()
            local_model = local_model.cuda()
            # local_model.cuda()

            for j in range(param['inner_epochs']):
                total_loss = 0.
                for batch_id, data in enumerate(dataloaders[i]):
                    x, y = data
                    x = x.cuda()
                    y = y.cuda()
                    output = local_model(x)
                    loss = loss_func(output, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

# 这里的dataloader我改过了，还能用.dataset吗 / float(len(dataloaders[i].dataset))
                logger.info('model {} | epoch {:3d} | internal_epoch {:3d} '
                                '| lr {:02.2f} | total_loss {:5.2f}'
                                            .format(i, epoch, j, param['lr'], total_loss))
            
            # local_model.cpu() 这个在第二轮就不对了
            for key, value in local_model.state_dict().items():
                value = value.cpu()
                if 'tracked' in key[-7:]:
                    continue
                try:
                    grad = value - global_model.state_dict()[key]
                except:
                    print (value.device, global_model.state_dict()[key].device)
                    print (key)
                    print ("第一处")
                    exit()
                # if (key == "bn1.num_batches_tracked"):
                #     print (value.cpu(), value.cpu().dtype)
                #     print (global_model.state_dict()[key], global_model.state_dict()[key].dtype)
                #     print ()
                # exit()
                try:
                    grad = grad / (param['num_train_parties'])
                    weight_accumulator[key] += grad
                except:
                    print (key, value)
                    print (global_model.state_dict()[key])
                    print (value.dtype, global_model.state_dict()[key].dtype)
                    exit()

    return weight_accumulator


def fedavg(global_model, weight_accumulator, param):
    for key in global_model.state_dict().keys():
        if 'tracked' in key[-7:]:
            continue
        global_model.state_dict()[key] += weight_accumulator[key] * param['eta']

    return global_model


def test(global_model, dataloader, epoch, param):
    loss_func = nn.CrossEntropyLoss()
    global_model = global_model.cuda()
    global_model.eval()
    total_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_id, data in enumerate(dataloader):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            # 记得看一下是否到了gpu上
            output = global_model(x)
            loss = loss_func(output, y)
            total_loss += loss.item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
        
        acc = 100.0 * (float(correct) / float(len(dataloader.dataset)))
        logger.info('global_model | epoch {:3d} '
                '| acc {:02.2f} | avg_loss {:5.2f}'
                            .format(epoch, acc, total_loss / float(len(dataloader.dataset))))
        

if __name__ == "__main__":
    # 配置logging
    logger = logging.getLogger("FL")
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.INFO)


    # 导入参数
    parser = argparse.ArgumentParser(description = "FL")
    parser.add_argument("--param", type = str, default = "param.yaml", help = "The location of parameters that FL will use")
    args = parser.parse_args()
    
    with open(args.param, "r") as f:
        param = yaml.load(f, Loader = yaml.FullLoader)
        logger.info("Finish the loading of parameters")
    

    # 配置helper，包括构建网络、加载数据
    helper = Helper(param)
    helper.load_data()


    # 训练模型, 测试模型
    logger.info("start the training")
    epochs = param['epochs']
    for epoch in range(epochs):
        weight_accumulator = local_train(helper.local_model, helper.global_model, helper.train_dataloaders, epoch, param)
        helper.global_model = fedavg(helper.global_model, weight_accumulator, param)
        test(helper.global_model, helper.test_dataloader, epoch, param)
