import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10

from model import vit_base_patch16_224

import os,math,argparse
from tqdm import tqdm
import sys

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        loss = loss_function(pred, labels.to(device))
        loss.backward()

        # eq逐行比较
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_loss += loss.detach()

        data_loader.desc = f"[train epoch {epoch}] " \
                           f"loss: {accu_loss.item() / (step + 1):.3f}," \
                           f" acc: {accu_num.item() / sample_num:.3f}"

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = f"[valid epoch {epoch}] " \
                           f"loss: {accu_loss.item() / (step + 1):.3f}," \
                           f" acc: {accu_num.item() / sample_num:.3f}"

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def main(args):

    device = args.device if torch.cuda.is_available() else 'cpu'

    if not os.path.exists('weights'):
        os.mkdir('weights')

    writer = SummaryWriter()

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }

    train_dataset = CIFAR10(root=args.data_path,
                            train=True,download=False,
                            transforms=data_transform['train'])
    val_dataset = CIFAR10(root=args.data_path,
                            train=False,download=False,
                            transforms=data_transform['val'])

    batch_size = args.batch_size
    num_of_worker = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(num_of_worker))

    train_loader = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_of_worker)

    val_loader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_of_worker)

    model = vit_base_patch16_224().to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."

        ckpt = torch.load(args.weights,map_location=device)
        del_keys = ['head.weight','head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del ckpt[k]
        print(model.load_state_dict(ckpt, strict=False))

    if args.freeze_layers:
        for name , para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad = False
            else:
                print(f"training {name}")

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters,lr=args.lr,momentum=0.9,weight_decay=1e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
            # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        writer.add_scalar(tags[0], train_loss, epoch)
        writer.add_scalar(tags[1], train_acc, epoch)
        writer.add_scalar(tags[2], val_loss, epoch)
        writer.add_scalar(tags[3], val_acc, epoch)
        writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data_path',type=str,default='data/cifar10')
    parser.add_argument('--weights',type=str,default='weights/jx_vit_base_p16_224-80ecf9dd.pth')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)