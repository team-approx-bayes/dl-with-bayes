To run training LeNet-5 for CIFAR-10 classification
```bash
python main.py --config <path/to/config> --download
```
| optimizer | dataset | architecture | config file path |
| --- | --- | --- | --- |
| [Adam](https://arxiv.org/abs/1412.6980) | CIFAR-10 | LeNet-5 | [configs/cifar10/lenet_adam.json](./configs/cifar10/lenet_adam.json) |
| [K-FAC](https://arxiv.org/abs/1503.05671)| CIFAR-10 | LeNet-5 | [configs/cifar10/lenet_kfac.json](./configs/cifar10/lenet_kfac.json) |
| [Noisy K-FAC](https://arxiv.org/abs/1712.02390)| CIFAR-10 | LeNet-5 | [configs/cifar10/lenet_noisykfac.json](./configs/cifar10/lenet_noisykfac.json) |
| [VOGN](https://arxiv.org/abs/1806.04854)| CIFAR-10 | LeNet-5 + BatchNorm | [configs/cifar10/lenet_vogn.json](./configs/cifar10/lenet_vogn.json) |
