import os
import argparse
import inspect

import imageio
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchsso

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--n_samples', type=int, default=100,
                        help='number of samples')
    parser.add_argument('--centers', type=int, default=5,
                        help='number of clusters')
    parser.add_argument('--random_state', type=int, default=5,
                        help='random seed for data creation')
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='input batch size for training')
    parser.add_argument('--plot_interval', type=int, default=20,
                        help='interval iterations to plot decision boundary')
    # Options
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_file_name', type=str, default='log',
                        help='log file name')
    parser.add_argument('--fig_dir', type=str, default='tmp',
                        help='directory to keep tmp figures')
    parser.add_argument('--keep_figures', action='store_true', default=False,
                        help='whether keep tmp figures after creating gif')
    parser.add_argument('--out', type=str, default='boundary.gif',
                        help='output gif file')

    args = parser.parse_args()

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Generate a dataset
    n_samples = args.n_samples
    centers = args.centers
    random_state = args.random_state

    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=centers, random_state=random_state)
    y[y < int(centers) / 2] = 0
    y[y >= int(centers) / 2] = 1

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    data_meshgrid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.float).to(device)

    X_tensor = torch.from_numpy(X).type(torch.float)
    y_tensor = torch.from_numpy(y).type(torch.float)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Model arguments
    model_kwargs = dict(input_size=2, output_size=None, hidden_sizes=[128])

    model1 = MLP(**model_kwargs)
    model1 = model1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters())

    model2 = pickle.loads(pickle.dumps(model1))  # create a clone
    model2 = model2.to(device)
    curv_kwargs = dict(ema_decay=0.01, damping=1e-7)
    optim_kwargs = dict(dataset_size=len(train_loader.dataset),
                        curv_type='Cov', curv_shapes={'Linear': 'Diag'}, curv_kwargs=curv_kwargs,
                        kl_weighting=1, warmup_kl_weighting_init=0.01, warmup_kl_weighting_steps=1000,
                        grad_ema_decay=0.1, num_mc_samples=50, val_num_mc_samples=20)
    optimizer2 = torchsso.optim.VIOptimizer(model2, **optim_kwargs)

    def get_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

    # Show all config
    print('===========================')
    print(f'model class: {model1.__class__}')
    print(f'model args: {model_kwargs}')
    for key, val in vars(args).items():
        print(f'{key}: {val}')
    print('---------------------------')
    print(f'optim1 class: {optimizer1.__class__}')
    print(f'optim1 args: {get_default_args(optimizer1.__init__)}')
    print('---------------------------')
    print(f'optim2 class: {optimizer2.__class__}')
    kwargs = get_default_args(optimizer2.__init__)
    kwargs.update(optim_kwargs)
    print(f'optim2 args: {kwargs}')
    print('===========================')

    figpaths = []
    i = 0  # iteration

    # Run training
    for epoch in range(args.epochs):

        model1.train()
        model2.train()

        for data, target in train_loader:

            data, target = data.to(device), target.to(device)

            def closure1():
                optimizer1.zero_grad()
                output = model1(data)
                loss = F.binary_cross_entropy_with_logits(output, target)
                loss.backward()
                return loss

            def closure2():
                optimizer2.zero_grad()
                output = model2(data)
                loss = F.binary_cross_entropy_with_logits(output, target)
                loss.backward()
                return loss, output

            loss1 = optimizer1.step(closure1)
            loss2, _ = optimizer2.step(closure2)

            if (i + 1) % args.plot_interval == 0:
                # Setup figure
                fig = plt.figure()
                plt.xlabel('Input 1')
                plt.ylabel('Input 2')
                plt.title(f'Iteration {i+1}')

                model1.eval()
                model2.eval()

                # (Adam)
                pred = torch.round(torch.sigmoid(model1(data_meshgrid)))
                pred = pred.detach().cpu().numpy().reshape(xx.shape)
                plot = plt.contour(xx, yy, pred, colors=['blue'], linewidths=[2])
                plot.collections[len(plot.collections)//2].set_label('Adam')

                # (VOGN) get MC samples
                _, probs = optimizer2.prediction(data_meshgrid, keep_probs=True)
                preds = [torch.round(p).detach().cpu().numpy().reshape(xx.shape) for p in probs]
                for pred in preds:
                    plt.contour(xx, yy, pred, colors=['red'], alpha=0.01)

                # (VOGN) get mean prediction
                prob = optimizer2.prediction(data_meshgrid, mc=0)
                pred = torch.round(prob).detach().cpu().numpy().reshape(xx.shape)
                plot = plt.contour(xx, yy, pred, colors=['red'], linewidths=[2])
                plot.collections[len(plot.collections)//2].set_label('VOGN')

                # plot samples
                for label, marker, color in zip([0, 1], ['o', 's'], ['white', 'gray']):
                    _X = X[y == label]
                    plt.scatter(_X[:, 0], _X[:, 1], s=80, c=color, edgecolors='black', marker=marker)

                # save tmp figure
                plt.grid(linestyle='--')
                plt.legend(loc='lower right')
                plt.tight_layout()
                figname = f'iteration{i+1}.png'
                figpath = os.path.join(args.fig_dir, figname)
                if not os.path.isdir(args.fig_dir):
                    os.makedirs(args.fig_dir)
                fig.savefig(figpath)
                plt.close(fig)
                figpaths.append(figpath)

            i += 1

        print(f'Train Epoch: {epoch+1}\tLoss(Adam): {loss1:.6f} Loss(VOGN): {loss2:.6f}')

    # Create GIF from temp figures
    images = []
    for figpath in figpaths:
        images.append(imageio.imread(figpath))
        if not args.keep_figures:
            os.remove(figpath)
    imageio.mimsave(args.out, images, fps=1)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, act_func="relu"):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if output_size is not None:
            self.output_size = output_size
            self.squeeze_output = False
        else:
            self.output_size = 1
            self.squeeze_output = True

        # Set activation function
        if act_func == "relu":
            self.act = F.relu
        elif act_func == "tanh":
            self.act = F.tanh
        elif act_func == "sigmoid":
            self.act = torch.sigmoid
        else:
            raise ValueError(f'Invalid activation function: {act_func}')

        # Define layers
        if hidden_sizes is None:
            # Linear model
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            features = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            self.hidden_layers = nn.ModuleList([nn.Linear(in_features, out_features) for in_features, out_features in features])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        h = x
        for layer in self.hidden_layers:
            h = self.act(layer(h))

        out = self.output_layer(h)
        if self.squeeze_output:
            out = torch.squeeze(out).view([-1])

        return out


if __name__ == '__main__':
    main()
