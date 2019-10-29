To run training on CIFAR-10/100 with multiple GPUs

```bash
mpirun -np <number of GPUs> python main.py \
--dist_init_method <init_method> \
--download \
--config <path/to/config>  
```

To run training on ImageNet with multiple GPUs

```bash
mpirun -np <number of GPUs> python main.py \
--train_root <path/to/train_data> \
--val_root <path/to/val_data> \
--dist_init_method <init_method> \
--config <path/to/config> 
```
For `init_method`, refer the [PyTorch tutorial for distirubted applications](https://pytorch.org/tutorials/intermediate/dist_tuto.html).

| optimizer | dataset | architecture | GPUs | config file path |
| --- | --- | --- | --- | --- |
| [Adam](https://arxiv.org/abs/1412.6980) | ImageNet | ResNet-18 | 128 | [configs/imagenet/resnet18_adam_bs4k_128gpu.json](./configs/imagenet/resnet18_adam_bs4k_128gpu.json) |
| [K-FAC](https://arxiv.org/abs/1503.05671) | ImageNet | ResNet-18 | 4 | [configs/imagenet/resnet18_kfac_bs4k_4gpu.json](./classification/configs/imagenet/resnet18_kfac_bs4k_4gpu.json) |
| [K-FAC](https://arxiv.org/abs/1503.05671)| ImageNet | ResNet-18 | 128 | [configs/imagenet/resnet18_kfac_bs4k_128gpu.json](./configs/imagenet/resnet18_kfac_bs4k_128gpu.json) |
| [Noisy K-FAC](https://arxiv.org/abs/1712.02390)| ImageNet | ResNet-18 | 128 | [configs/imagenet/resnet18_noisykfac_bs4k_128gpu.json](./configs/imagenet/resnet18_noisykfac_bs4k_128gpu.json) |
| [VOGN](https://arxiv.org/abs/1806.04854)| ImageNet | ResNet-18 | 128 | [configs/imagenet/resnet18_vogn_bs4k_128gpu.json](./configs/imagenet/resnet18_vogn_bs4k_128gpu.json) |

- NOTE:
  - You need to run with `N` GPUs when you use `*{N}gpu.json` config file.
  - You need to set `--acc_steps` (or `"acc_steps"` in json config) to run with limited number of GPUs as below:
    - Mini-batch size (bs) = {examples per GPU} x {# GPUs} x {acc_steps}
    - Ex) 4096 (4k) = 32 x 8 x 16
  - The gradients of loss and the curvature are accumulated for `acc_steps` to build pseudo mini-batch size. 

Visit [configs](./configs) for other architecture, dataset, optimizer, number of GPUs.
