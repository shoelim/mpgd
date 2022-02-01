import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="ConvMixer")

parser.add_argument('--batch-size', default=64, type=int) #original: 512
parser.add_argument('--scale', default=0.75, type=float)
parser.add_argument('--reprob', default=0.25, type=float)
parser.add_argument('--ra-m', default=8, type=int)
parser.add_argument('--ra-n', default=1, type=int)
parser.add_argument('--jitter', default=0.1, type=float)

parser.add_argument('--hdim', default=256, type=int) 
parser.add_argument('--depth', default=10, type=int) #original: 8
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)

parser.add_argument('--wd', default=0.01, type=float)
parser.add_argument('--clip-norm', action='store_true')
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--lr-max', default=0.01, type=float)
parser.add_argument('--workers', default=4, type=int) #original: 2
parser.add_argument('--seed', default=1, type=int) #original: not specified
parser.add_argument('--option', default=0, type=int) #original: none

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def PM_map(x, gamma=0.7):
  if x < 0.5:
    return x*(1+(2**gamma)*(x**gamma))
  else:
    return 2*x-1

def Thaler_map(x, gamma=0.6):
  return (x*(1+(x/(1+x))**(gamma-1) - x**(gamma-1))**(1/(1-gamma))) % 1


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(args.scale, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=args.ra_n, magnitude=args.ra_m),
    transforms.ColorJitter(args.jitter, args.jitter, args.jitter),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
    transforms.RandomErasing(p=args.reprob)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)


model = ConvMixer(args.hdim, args.depth, patch_size=args.psize, kernel_size=args.conv_ks, n_classes=10)
model = nn.DataParallel(model).cuda()


#lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], 
#                                  [0, args.lr_max, args.lr_max/20.0, 0])[0]

opt = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.wd)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

if args.option == 1:
    gamma = 0.7
    r = 0.1 #(0.01)**gamma
elif args.option == 2:
    gamma = 0.6
    r = 0.1 #(0.01)**gamma

#count = 0

for epoch in range(args.epochs):
    start = time.time()
    train_loss, train_acc, n = 0, 0, 0
    for i, (X, y) in enumerate(trainloader):
        model.train()
        X, y = X.cuda(), y.cuda()

        #lr = lr_schedule(epoch + (i + 1)/len(trainloader)) 
        lr = 0.001 #/(1+0.02*epoch) #0.002/(1+0.005*epoch)
        opt.param_groups[0].update(lr=lr)

        opt.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(X)
            
            if args.option == 0:
                loss = criterion(output, y)
            elif args.option == -1:
                reg = 0.0
                for param in model.parameters():
                    reg += 0.001 * (param ** 2).sum()
                loss = criterion(output, y) + reg
            else:
                reg = 0.0
                for param in model.parameters():
                    reg += 0.01 * (param ** 2).sum()
                loss = criterion(output, y) + r * reg #/(1+0.02*count)
                if args.option == 1:
                    r = PM_map(r,gamma=gamma)
                elif args.option == 2:
                    r = Thaler_map(r,gamma=gamma)
                else:
                    print('Invalid option!')
                #count += 1  


        scaler.scale(loss).backward()
        if args.clip_norm:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
        
    model.eval()
    test_acc, m = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(testloader):
            X, y = X.cuda(), y.cuda()
            with torch.cuda.amp.autocast():
                output = model(X)
            test_acc += (output.max(1)[1] == y).sum().item()
            m += y.size(0)

    #print(epoch, test_acc)
    print(f'[{args.name}] Epoch: {epoch} | Train Acc: {train_acc/n:.4f}, Test Acc: {test_acc/m:.4f}, Time: {time.time() - start:.1f}, lr: {lr:.6f}')
