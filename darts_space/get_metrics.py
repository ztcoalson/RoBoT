import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import pickle
import copy
import json

from tqdm import tqdm

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

from scipy.special import softmax
from collections import defaultdict

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets as dset
from torch.utils.data import Subset

from bayes_opt import BayesianOptimization

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from darts_space import utils
from darts_space.genotypes import *

from foresight.pruners.measures import fisher, grad_norm, grasp, snip, synflow, jacov

sys.path.append("../poisons/")
from poisons import LabelFlippingPoisoningDataset, CleanLabelPoisoningDataset
from poisons_utils import imshow

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--arch_path', type=str, default='data/sampled_archs.p', help='location of the data corpus')
parser.add_argument('--no_search', action='store_true',default=False, help='only apply sampling')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for search')
parser.add_argument('--batch_size', type=int, default=576, help='batch size')
parser.add_argument('--metric_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=5e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--n_sample', type=int, default=100, help='number of genotypes to evaluate')

parser.add_argument('--scale', type=float, default=1e2, help="")

parser.add_argument('--total_iters', type=int, default=25, help='pytorch manual seed')
parser.add_argument('--init_portion', type=float, default=0.25, help='pytorch manual seed')
parser.add_argument('--acq', type=str, default='ucb',help='choice of bo acquisition function, [ucb, ei, poi]')

# poisoning args
parser.add_argument('--poisons_type', type=str, choices=['clean_label', 'label_flip', 'none'], default='none')
parser.add_argument('--poisons_path', type=str, default=None)

args = parser.parse_args()

args.cutout = False
args.auxiliary = False

if args.dataset in ['cifar10', 'mnist']:
    NUM_CLASSES = 10
    from darts_space.model import NetworkCIFAR as Network
elif args.dataset == 'cifar100':
    NUM_CLASSES = 100
    from darts_space.model import NetworkCIFAR as Network
elif args.dataset == 'imagenet':
    NUM_CLASSES = 1000
    from darts_space.model import NetworkImageNet as Network
else:
    raise ValueError('Donot support dataset %s' % args.dataset)

def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()

def compute_metrics(net, inputs, targets):
    metric_list = {}
    metric_list['fisher'] = sum_arr(fisher.compute_fisher_per_weight(copy.deepcopy(net).cuda(), inputs, targets, F.cross_entropy, "channel"))
    metric_list['grad_norm'] = sum_arr(grad_norm.get_grad_norm_arr(copy.deepcopy(net).cuda(), inputs, targets, F.cross_entropy))
    metric_list['snip'] = sum_arr(snip.compute_snip_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param", F.cross_entropy))
    metric_list['synflow'] = sum_arr(synflow.compute_synflow_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param"))
    metric_list['jacov'] = jacov.compute_jacob_cov(copy.deepcopy(net).cuda(), inputs, targets)
    metric_list['grasp'] = sum_arr(grasp.compute_grasp_per_weight(copy.deepcopy(net).cuda(), inputs, targets, "param", F.cross_entropy))
    return metric_list

def genotype(weights, steps=4, multiplier=4):
    def _parse(weights):
        gene = []
        n = 2
        start = 0
        for i in range(steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(
                W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
        return gene
        
    gene_normal = _parse(softmax(weights[0], axis=-1))
    gene_reduce = _parse(softmax(weights[1], axis=-1))

    concat = range(2+steps-multiplier, steps+2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

def percentage_change(old, new):
    return ((new - old) / abs(old)) * 100

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print('gpu device = %d' % args.gpu)
    print("args = %s", args)

    # for poisoning attacks
    train_kwargs = {
        'root': args.data,
        'train': True,
        'download': True,
        'transform': None,
    }

    # sample 1000 random indices from the training set
    indices = np.random.choice(50000, 1000, replace=False)

    _, valid_transform = eval("utils._data_transforms_%s" % args.dataset)(args)
    clean_train_data = eval("dset.%s" % args.dataset.upper())(
            root=args.data, train=True, download=True, transform=valid_transform)
    clean_data = Subset(clean_train_data, indices)
    clean_queue = torch.utils.data.DataLoader(
        clean_data, batch_size=args.metric_batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    rlf_train_data = LabelFlippingPoisoningDataset("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/rlf/rlf-cifar10-50.0%.pth", valid_transform, train_kwargs)
    rlf_data = Subset(rlf_train_data, indices)
    rlf_queue = torch.utils.data.DataLoader(
        rlf_data, batch_size=args.metric_batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )
    
    clf_train_data = LabelFlippingPoisoningDataset("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/smart-lf/smart-lf-resnet18-cifar10-50.0%.pth", valid_transform, train_kwargs)
    clf_data = Subset(clf_train_data, indices)
    clf_queue = torch.utils.data.DataLoader(
        clf_data, batch_size=args.metric_batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )
    
    noise_train_data = CleanLabelPoisoningDataset("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/poisons/noise/noise-cifar10-50.0%.pth", valid_transform, train_kwargs)
    noise_data = Subset(noise_train_data, indices)
    noise_queue = torch.utils.data.DataLoader(
        noise_data, batch_size=args.metric_batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )
    
    gc_train_data = CleanLabelPoisoningDataset("/nfs/hpc/share/coalsonz/NAS-Poisoning-Dev/poisons/gc_runs/robot/gc/robot-eps=0.5-gc-d-darts-50.0%-20250115-201953/poisons.pth", valid_transform, train_kwargs)
    gc_data = Subset(gc_train_data, indices)
    gc_queue = torch.utils.data.DataLoader(
        gc_data, batch_size=args.metric_batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )

    assert len(clean_queue) == len(rlf_queue) == len(clf_queue) == len(noise_queue) == len(gc_queue)

    metric_names = ['grad_norm', 'snip', 'grasp', 'fisher', 'synflow', 'jacov']

    size=[14 * 2, 7]
    new_weights = [np.random.random_sample(size) for _ in range(args.n_sample)]
    new_genos = [genotype(w.reshape(2, -1, size[-1])) for w in new_weights]

    results = {}
    for geno_idx, geno in tqdm(enumerate(new_genos), desc="Computing metrics", total=args.n_sample):
        arch_results = {
            'clean': {},
            'rlf': {},
            'clf': {},
            'noise': {},
            'gc': {}
        }

        model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, geno).cuda()
        model.drop_path_prob = 0

        for inputs, targets in clean_queue:
            inputs, targets = inputs.cuda(), targets.cuda()
            metric_list = compute_metrics(model, inputs, targets)
            for k in metric_list.keys():
                arch_results['clean'][k] = arch_results['clean'].get(k, 0) + metric_list[k]
        
        for inputs, targets in rlf_queue:
            inputs, targets = inputs.cuda(), targets.cuda()
            metric_list = compute_metrics(model, inputs, targets)
            for k in metric_list.keys():
                arch_results['rlf'][k] = arch_results['rlf'].get(k, 0) + metric_list[k]
        
        for inputs, targets in clf_queue:
            inputs, targets = inputs.cuda(), targets.cuda()
            metric_list = compute_metrics(model, inputs, targets)
            for k in metric_list.keys():
                arch_results['clf'][k] = arch_results['clf'].get(k, 0) + metric_list[k]
        
        for inputs, targets in noise_queue:
            inputs, targets = inputs.cuda(), targets.cuda()
            metric_list = compute_metrics(model, inputs, targets)
            for k in metric_list.keys():
                arch_results['noise'][k] = arch_results['noise'].get(k, 0) + metric_list[k]
        
        for inputs, targets in gc_queue:
            inputs, targets = inputs.cuda(), targets.cuda()
            metric_list = compute_metrics(model, inputs, targets)
            for k in metric_list.keys():
                arch_results['gc'][k] = arch_results['gc'].get(k, 0) + metric_list[k]

        baseline = arch_results['clean']
        for atk in ['clean', 'rlf', 'clf', 'noise', 'gc']:
            for k in metric_names:
                if atk == 'clean':
                    diff = baseline[k] / len(clean_queue)
                else:
                    diff = percentage_change(baseline[k] / len(clean_queue), arch_results[atk][k] / len(clean_queue))

                if geno_idx not in results:
                    results[geno_idx] = {}
                if atk not in results[geno_idx]:
                    results[geno_idx][atk] = {}
                
                results[geno_idx][atk][k] = diff

    # process results
    final_results = {
        'clean': {},
        'rlf': {},
        'clf': {},
        'noise': {},
        'gc': {}
    }
    for atk in ['clean', 'rlf', 'clf', 'noise', 'gc']:
        for k in metric_names:
            metric_results = []
            for geno_idx in results.keys():
                metric_results.append(results[geno_idx][atk][k])
            
            final_results[atk][k] = {
                'mean': np.mean(metric_results),
                'std': np.std(metric_results)
            }
    
    with open(f"metric_sens_results/metrics_{args.n_sample}.json", "w") as f:
        json.dump(final_results, f, indent=4)


if __name__ == '__main__':
    main()