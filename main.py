import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import random
import yaml
import logging
import seaborn as sns
from utils import Helper

def local_train(local_model, global_model, dataloaders, epoch, param):
    # 所有客户端同时做local training，然后将权重求和返回
    weight_accumulator = dict()
    for key, value in global_model.state_dict().items():
        weight_accumulator[key] = torch.zeros_like(value)

    optimizer = torch.optim.Adam(local_model.parameters(), lr = param['lr'])
    loss_func = nn.CrossEntropyLoss()

    train_list = random.sample(range(param['parties']), param['num_train_parties'])
    logger.debug(f"the parties which are chosen: {train_list}")
    for i in range(param['parties']):
        if i in train_list:
            local_model.copy_params(global_model.state_dict())
            local_model.train()

            for j in range(param['inner_epochs']):
                correct = 0
                total = 0
                total_loss = 0
                for batch_id, (inputs, labels) in enumerate(dataloaders[i]):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = local_model(inputs)
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total += outputs.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()

                logger.debug('model {} | epoch {:3d} | internal_epoch {:3d} | '
                             'lr {:02.2f} | acc {:02.2f} | total_loss {:5.2f}\n'
                                            .format(i, epoch, j, param['lr'], correct/total, total_loss))
            
            for key, value in local_model.state_dict().items():
                if 'tracked' in key[-7:]:
                    continue
                grad = value - global_model.state_dict()[key]
                grad = grad * len(dataloaders[i].sampler)
                sum_s = sum([len(dataloaders[i].sampler) for i in train_list])
                grad /= sum_s
                weight_accumulator[key] += grad

    return weight_accumulator


def fedavg(global_model, weight_accumulator, param):
    for key in global_model.state_dict().keys():
        if 'tracked' in key[-7:]:
            continue
        global_model.state_dict()[key] += weight_accumulator[key] * param['eta']

    return global_model


def test(model, dataloader):
    loss_func = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            total_loss += loss.item()
            total += outputs.size(0)
            _, predicted = outputs.max(1)  
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    logger.info('global_model | epoch {:3d} | acc {:02.2f} | total_loss {:5.2f}\n'.format(epoch, acc, total_loss))
        

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

        test(helper.global_model, helper.test_dataloader)

    
    # 可视化
    