import torch
import torch.backends.cudnn as cudnn
from datasets import PascalVOCDataset
from model import SSD300, MultiBoxLoss
import utils

import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

# The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations.
# Training on a RTX 3090 cost about 10 hours with average GPU usage about 70%.
# If your GPU has less vram, try to decrease the batch size, and change the iterations and decay_lr_at accordingly.

# Dataset and DataLoader
keep_difficult = True  # use objects considered difficult to detect?
num_workers = 8  # number of workers for loading data in the DataLoader
batch_size = 32

# Optimizer
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate

# Training
checkpoint_path = 'ssd300.pt'  # path to model checkpoint, None if none
iterations = 120000
print_freq = 10  # print training status every _ batches

# clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32)
# you will recognize it by a sorting error in the MuliBox loss calculation
grad_clip = None


def train(dataloader, model, criterion, optimizer, epoch):
    """
    One epoch's traning.

    :param dataloader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    start = time.time()

    # Batches
    for i, (images, boxes, labels) in enumerate(dataloader):

        # Move to default device
        images = images.to(device)  # (batch_size(N), 3, 300, 300)
        boxes = [bs.to(device) for bs in boxes]
        labels = [ls.to(device) for ls in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Barkward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(dataloader), batch_time=batch_time, loss=losses))


def main():
    # Prepare train dataset and dataloader
    train_ds = PascalVOCDataset('./data', 'TRAIN', keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=train_ds.collate_fn,  # note that we're passing the collate function here
                                               num_workers=num_workers,
                                               pin_memory=True)
    n_classes = len(train_ds.label_map())
    start_epoch = 0

    # Initialize model
    model = SSD300(n_classes=n_classes)

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Load checkpoint if existed
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print('Load checkpoint from epoch %d.\n' % checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    model.to(device)
    model.train()

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    epochs = iterations // (len(train_ds) // batch_size)
    decay_lr_at_epochs = [it // (len(train_ds) // batch_size) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at_epochs:
            utils.adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader, model, criterion, optimizer, epoch)

        # Save checkpoint
        utils.save_checkpoint(checkpoint_path, model, optimizer, epoch)


if __name__ == '__main__':
    main()
