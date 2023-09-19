import os
import warnings

import torch
from loss.label_smoothing import LabelSmoothingCrossEntropy
 
import sys
from importlib import import_module
import numpy as np
import random
torch.manual_seed(199656)  # 为CPU设置随机种子
np.random.seed(199656)  # Numpy module.
random.seed(199656)
torch.cuda.manual_seed(199656)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(199656)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from Dataset import SkeletonDataset


from Tools.common import get_cuda_id
cuda_id = get_cuda_id()
print('run on cuda:',cuda_id)
torch.cuda.set_device(cuda_id)
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

from Tools.common import load_yaml
args = load_yaml(f'configs/{dataset}_MSGCN.yaml')
dataset = 'lara'
net_name = 'TPM'
smooth_period = 4
batch_size = args.batch_size
learning_rate = args.learning_rate
if not os.path.exists(f'ckpts/{dataset}/{net_name}'):
    os.makedirs(f'ckpts/{dataset}/{net_name}')
trainset = SkeletonDataset(args, mode='train')
train_dataloader = DataLoader(dataset=trainset, batch_size=batch_size,
                            collate_fn=trainset.collate_fn,
                            shuffle=True, num_workers=8, drop_last=False)
testset = SkeletonDataset(args, mode='test')
test_dataloader = DataLoader(dataset=testset, batch_size= 2 * batch_size,
                            collate_fn=testset.collate_fn,
                            shuffle=False, num_workers=8, drop_last=False)
ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
mse = nn.MSELoss(reduction='none')
smloss = LabelSmoothingCrossEntropy()
Model = getattr(import_module(f'models.{net_name}'),'Model')
net = Model(args).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[20, 40, 60, 80, 120], gamma=0.8)
from Tools.metric import ScoreMeter
def train():
    net.train()
    for data in tqdm(train_dataloader):
        features = data['feature'].cuda()
        label = data['annotation'].cuda()
        mask = data['mask'].cuda()
        optimizer.zero_grad()
        output_list = net(features, mask)
        [l0,l1, l2, l3, l4, l5, avg_pred, smooth_label] = output_list
        if e % smooth_period != 0:
            log_avg_pred = torch.log(avg_pred + 1e-10) # to prevent -inf
            loss = ce(log_avg_pred.transpose(2, 1).contiguous().view(-1, args.num_classes), label.view(-1))
        else:
            loss = 0
            for idx, logit in enumerate([l0,l1, l2, l3, l4, l5]):
                dim = logit.shape[2]
                logit = logit.transpose(2, 1).contiguous()

                _sm_label = F.interpolate(smooth_label[idx].float(), size=dim, mode='linear').squeeze()
                _mask = F.interpolate(mask, size=dim)

                _loss = smloss(logit.view(-1, args.num_classes),
                               _sm_label.transpose(2, 1).contiguous().view(-1, args.num_classes),
                               _mask.transpose(2, 1).contiguous().view(-1, args.num_classes))
                loss += _loss
        loss.backward()
        optimizer.step()

def eval():
    net.eval()
    evaled = []

    scoreMeter = ScoreMeter(
        iou_thresholds=(0.1, 0.25, 0.5),
        n_classes=args.num_classes
    )

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            x = data['feature'].cuda()
            t = data['annotation'].cuda()
            mask = data['mask'].cuda()
            names = data['names']
            avg_pred = net(x, mask)[6]
            batch_size = x.shape[0]
            for i in range(batch_size):
                name = names[i]
                if evaled.__contains__(name):
                    continue
                else:
                    evaled.append(name)
                    m = mask[i]
                    cnt = int(torch.sum(m[0]))
                    prob = avg_pred[i].detach()[:, :cnt]
                    target = t[i][:cnt].cpu().numpy()
                    prediction = torch.nn.Softmax(dim=0)(prob)
                    predicted = torch.max(prediction, dim=0)[1]
                    predicted = F.interpolate(predicted.unsqueeze(0).unsqueeze(0).float(), size=target.shape[0]).squeeze().long()
                    predicted = predicted.cpu().data.numpy()
                    scoreMeter.update(predicted, target)

        scores = scoreMeter.get_scores()
        result_dict = {'*acc': f'*{round(scores[0], 2)}*',
             '*edit*': f'*{round(scores[1], 2)}*',
             '*f1*': f'*{round(scores[2][0], 2)}*',
             '*f2*': f'*{round(scores[2][1], 2)}*',
             '*f3*': f'*{round(scores[2][2], 2)}*'}
        torch.save(net.state_dict(), f'ckpts/{dataset}/{net_name}/{e :02d}.pt')
        return result_dict


results = []
for epoch in range(args.num_epochs):
    print('epoch', epoch, ':')
    e = epoch + 1
    train()

    if e % args.eval_period == 0:
        result = eval()
        pprint (result)
