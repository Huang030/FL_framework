import torch
from torchvision import datasets, transforms
from collections import defaultdict
import random
import numpy as np
import models
import yaml

class Helper:
    def __init__(self, param):
        self.local_model = None
        self.global_model = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_dataloaders = dict()
        self.test_dataloader = None
        self.param = param
        
    def load_data(self):
        # 加载数据集
        train_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
        test_transform = transforms.Compose([transforms.ToTensor()])
        if self.param['dataset'] == "cifar10":
            self.train_dataset = datasets.CIFAR10(root = "~/data", train = True, transform = train_transform, download = True)
            self.test_dataset = datasets.CIFAR10(root = "~/data", train = False, transform = test_transform, download = True)

        # 加载模型 
        if self.param['model'] == "resnet":
            self.local_model = models.resnet.ResNet18(name = 'local_model')
            self.global_model = models.resnet.ResNet18(name = 'global_model')

        # 配置数据加载器
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, 
                                                           batch_size = self.param['batch_size'], 
                                                           shuffle = True)
        per_party_list = sample_dirichlet(self.train_dataset, self.param['parties'], self.param['alpha'])
        for party in range(self.param['parties']):
            sampler = torch.utils.data.SubsetRandomSampler(per_party_list[party])
            self.train_dataloaders[party] = torch.utils.data.DataLoader(self.train_dataset,
                                                                        batch_size = self.param['batch_size'], 
                                                                        sampler = sampler)


def sample_dirichlet(train_dataset, parties, alpha = 0.9):
    # 迪利克雷采样，用于load_data中训练集的数据加载器配置
    # 使用np.random.dirichlet生成每个参与者获得每张图片的概率，这里使用的方法是先打乱，然后直接用概率乘以总数，最后从数据集中取出图片
    """
        Input: Number of parties and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
    """
    classes = defaultdict(list)
    for index, data in enumerate(train_dataset):
        _, y = data
        classes[y].append(index)

    per_party_list = defaultdict(list)
    num_classes = len(classes.keys())
    for n in range(num_classes):
        random.shuffle(classes[n])
        sampled_probabilities = len(classes[n]) * np.random.dirichlet(np.array(parties * [alpha]))
        for party in range(parties):
            num_imgs = int(round(sampled_probabilities[party]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            per_party_list[party].extend(sampled_list)
            classes[n] = classes[n][min(len(classes[n]), num_imgs):]

    return per_party_list
    

if __name__ == "__main__":
    with open("param.yaml", "r") as f:
        param = yaml.load(f, Loader = yaml.FullLoader)
    helper = Helper(param)
    print (helper.param)
    helper.load_data()
    print (dir(helper))



    