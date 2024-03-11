# FL_framework

This repository implements the FedAvg algorithm on Pytorch.

## Prerequisites

- Python 3.6+
- PyTorch 1.0+

## Training

```
# Start training with:
python main.py

# You can manually modify the param.yaml file and then start training with:
python main.py --param param.yaml
```

## Notes

Currently, the code only implements versions of ResNet18+Cifar10. In the future, training processes for other models and datasets may be added.
