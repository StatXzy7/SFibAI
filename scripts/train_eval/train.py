import os
import time
import warnings
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import Config
from data.dataset import SchistosomiasisDataset
from data.transforms import ImageTransforms
from utils.metrics import validate, generate_soft_labels
from utils.visualization import plot_training_progress
from utils.models import create_model

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for liver fibrosis detection')
    
    # Training related parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument("--epochs", type=int, default=Config.EPOCHS,
                                help='Number of training epochs')
    training_group.add_argument("--bs", type=int, default=Config.BATCH_SIZE,
                                help='Batch size')
    training_group.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS,
                                help='Number of worker processes for data loading')
    training_group.add_argument("--ddp_enabled", type=bool, default=Config.DDP_ENABLED,
                                help='Whether to enable distributed training')
    training_group.add_argument("--device_id", type=int, default=Config.DEVICE_ID,
                                help='GPU device ID')
    
    # Optimizer and learning rate related parameters
    optimizer_group = parser.add_argument_group('Optimizer Parameters')
    optimizer_group.add_argument("--lr0", type=float, default=Config.LEARNING_RATE,
                                 help='Initial learning rate')
    optimizer_group.add_argument("--lr1", type=float, default=Config.MIN_LEARNING_RATE,
                                 help='Minimum learning rate')
    optimizer_group.add_argument("--scheduler", type=str, default=Config.SCHEDULER,
                                 choices=['cos', 'step'], help='Learning rate scheduler type')
    optimizer_group.add_argument("--tmax", type=int, default=Config.T_MAX,
                                 help='Learning rate scheduler period')
    
    # Model related parameters
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument("--num_classes", type=int, default=Config.NUM_CLASSES,
                             help='Number of classification classes')
    model_group.add_argument("--checkpoint_path", type=str, default=Config.CHECKPOINT_PATH,
                             help='Path to pretrained model')
    model_group.add_argument("--backbone",
                             type=str,
                             default=Config.BACKBONE,
                             choices=[
                                 "resnet50",
                                 "resnext50_32x4d",
                                 "mobilenet_v2",
                                 "mobilenet_v3_large",
                                 "densenet121",
                                 "efficientnet_b0",
                                 "efficientnet_b1",
                             ],
                             help="Backbone network model to use")
    
    # Data related parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument("--root_dirs", nargs='+', default=Config.ROOT_DIRS,
                           help='List of dataset root directories')
    data_group.add_argument("--shape", nargs='+', default=Config.IMAGE_SIZE,
                           help='Input image dimensions')
    
    # Loss function related parameters
    loss_group = parser.add_argument_group('Loss Function Parameters')
    loss_group.add_argument("--loss", type=str, default='hybrid',
                           choices=['kl', 'mse', 'boundary',  # Single losses
                                  'kl_mse', 'kl_boundary', 'mse_boundary',  # Double loss combinations
                                  'hybrid'],  # Complete hybrid loss
                           help='Loss function type')
    loss_group.add_argument("--alpha", type=float, default=1.0,
                           help='KL divergence loss weight, loss converges to ~0.06')
    loss_group.add_argument("--beta", type=float, default=0.02,
                           help='MSE loss weight, loss converges to ~1.45, 0.06/1.45=0.0414')
    loss_group.add_argument("--gamma", type=float, default=0.02,
                           help='Boundary penalty weight, loss converges to ~0.94, 0.06/0.94=0.0638')
    loss_group.add_argument("--std", type=float, default=1,
                           help='Standard deviation for soft label generation')
    loss_group.add_argument("--shape_param", type=float, default=1,
                           help='Distribution shape parameter')
    loss_group.add_argument("--scale_param", type=float, default=1,
                           help='Distribution scale parameter')
    
    # Save related parameters
    save_group = parser.add_argument_group('Save Parameters')
    save_group.add_argument("--save_root", type=str, default=Config.SAVE_ROOT,
                           help='Save root directory')
    save_group.add_argument("--save_interval", type=int, default=100,
                           help='Interval for printing training information (every N batches)')
    save_group.add_argument("--val_batch_size", type=int, default=100,
                           help='Batch size for validation')
    save_group.add_argument("--save_best_name", type=str, default=Config.SAVE_BEST_NAME,
                           help='Filename for saving best model')
    save_group.add_argument("--save_time_format", type=str, default=Config.SAVE_TIME_FORMAT,
                           help='Time format for save directory')
    
    parser.add_argument(
        "--crop_mode",
        type=str,
        default="mixed",
        choices=['none', 'fixed', 'random', 'mixed'],
        help="Crop mode: none-no cropping, fixed-fixed cropping, random-random cropping, mixed-mixed cropping (default)"
    )
    
    return parser.parse_args()

def setup_ddp(ddp_enabled=False, device_id=0):
    if ddp_enabled:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda", device_id)
    return local_rank, device

def setup_model(device, args):
    # Create model
    model = create_model(
        backbone=args.backbone,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint_path
    )
    model = model.to(device)
    return model

def setup_scheduler(optimizer, args):
    if args.scheduler == 'cos':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.tmax, eta_min=args.lr1)
    elif args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.tmax, gamma=0.6)

def setup_dataloaders(train_dataset, val_dataset, batch_size, ddp_enabled, num_workers):
    if ddp_enabled:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            sampler=train_sampler
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    return train_loader, val_loader

def calculate_loss(y_hat, y, loss_type, device, args):
    """Calculate different types of loss functions

    Args:
        y_hat: Model output
        y: Ground truth labels
        loss_type: Loss function type
        device: Computing device
        args: Training arguments
    """
    # KL divergence loss (local KL version)
    def get_kl_loss():
        soft_labels = generate_soft_labels(y, device=device)
        kl_crit = nn.KLDivLoss(reduction='none')
        
        # Calculate KL divergence
        kl_loss = kl_crit(F.log_softmax(y_hat, dim=1), soft_labels)
        
        # Select local range [-2, 2], ensure indices are within [0, 35]
        valid = (y.unsqueeze(1) + torch.arange(-2, 3).to(device)).clamp(0, 35)
        
        # Calculate KL divergence only within valid label range
        kl_loss_local = kl_loss.gather(1, valid)
        
        # Average local KL loss
        return kl_loss_local.sum() / kl_loss.size(0)

    # Mean Squared Error (MSE) loss
    def get_mse_loss():
        y_pred = torch.sum(y_hat.softmax(-1) * torch.arange(36).to(device), dim=1)
        return nn.MSELoss()(y_pred.float(), y.float())

    # Boundary penalty loss
    def get_boundary_loss():
        y_pred = torch.sum(y_hat.softmax(-1) * torch.arange(36).to(device), dim=1)
        y_pred_floor = torch.floor(y_pred)
        y_floor = torch.floor(y.float())
        boundary_mask = (y_pred_floor != y_floor)
        return (torch.abs(y_pred - y.float()) * boundary_mask.float()).mean()

    # Single loss functions
    if loss_type == 'kl' or loss_type == 'kl_p':  # Default using local KL
        return get_kl_loss()

    elif loss_type == 'mse':  # Only MSE
        return get_mse_loss()

    elif loss_type == 'boundary':  # Only boundary penalty
        return get_boundary_loss()

    # Double loss combinations
    elif loss_type == 'kl_mse':  # Local KL + MSE
        return args.alpha * get_kl_loss() + args.beta * get_mse_loss()

    elif loss_type == 'kl_boundary':  # Local KL + boundary penalty
        return args.alpha * get_kl_loss() + args.gamma * get_boundary_loss()

    elif loss_type == 'mse_boundary':  # MSE + boundary penalty
        return args.beta * get_mse_loss() + args.gamma * get_boundary_loss()

    # Complete hybrid loss function
    elif loss_type == 'hybrid':  # Local KL + MSE + boundary penalty
        loss_kl = get_kl_loss()
        loss_mse = get_mse_loss()
        loss_boundary = get_boundary_loss()

        return (args.alpha * loss_kl +
                args.beta * loss_mse +
                args.gamma * loss_boundary)

    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")

def train_epoch(model, train_loader, optimizer, scaler, device, local_rank, args):
    total_loss = 0
    
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        with autocast():
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            
            loss = calculate_loss(y_hat, y, args.loss, device, args)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if local_rank == 0 and (i+1) % args.save_interval == 0:
            batch_size = x.size(0)
            y_hat = y_hat.softmax(-1)
            y_hat = torch.sum(y_hat*torch.arange(36).to(device), dim=1)
            y_hat = y_hat.long()
            correct = (y_hat == y).sum().item()
            err = torch.abs(y_hat-y).sum().item()
            print(f'batch: {i+1}, loss: {loss.item():.3f}, '
                  f'accuracy: {correct/batch_size:.3f}, '
                  f'err: {err/batch_size:.3f}')
    
    return {'loss': total_loss / len(train_loader)}

def train(model, train_loader, val_loader, optimizer, scheduler, device, args, save_folder, local_rank):
    best_acc = 0
    losses, acces, lrs, errs = [], [], [], []
    scaler = GradScaler()
    
    for epoch in range(args.epochs):
        if local_rank == 0:
            print(f'------------- epoch: {epoch+1} -------------')
            print(f'lr: {scheduler.get_last_lr()}')
            
        train_metrics = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scaler, 
            device, 
            local_rank,
            args
        )
        
        scheduler.step()
        
        if local_rank == 0:
            acc, acc2, err = validate(
                model,
                val_loader,
                device,
                save_folder=save_folder,
                save_mat=True,
                epoch=epoch
            )
            
            losses.append(train_metrics['loss'])
            acces.append(acc2)
            errs.append(err)
            lrs.append(scheduler.get_last_lr()[0])
            
            if acc2 > best_acc:
                best_acc = acc2
                model_name = os.path.join(save_folder, args.save_best_name)
                torch.save(model.state_dict(), model_name)
            
            plot_training_progress(
                losses, acces, errs, lrs,
                save_folder,
                best_acc
            )

def main():
    args = parse_args()
    local_rank, device = setup_ddp(args.ddp_enabled, args.device_id)
    
    # Create save directory with loss type information
    cur_save_folder = os.path.join(
        args.save_root,
        f"{time.strftime(args.save_time_format)}_{args.backbone}_{args.loss}"
    )
    if local_rank == 0:
        os.makedirs(cur_save_folder, exist_ok=True)
        
    # Dataset and dataloader setup
    train_transforms = ImageTransforms(shape=args.shape, training=True)
    val_transforms = ImageTransforms(shape=args.shape, training=False)
    
    train_dataset = SchistosomiasisDataset(
        args.root_dirs,
        mode='train', 
        transform=train_transforms,
        crop_mode=args.crop_mode
    )
    val_dataset = SchistosomiasisDataset(
        args.root_dirs,
        mode='val', 
        transform=val_transforms,
        crop_mode=args.crop_mode
    )
    
    train_loader, val_loader = setup_dataloaders(
        train_dataset, 
        val_dataset,
        args.bs,
        args.ddp_enabled,
        args.num_workers
    )
    
    model = setup_model(device, args)
    if args.ddp_enabled:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': [nn.Parameter(torch.tensor([1.0, 3.0], device=device))], 'lr': 1e-3, 'name': 'loss_weights'}
    ], lr=args.lr0)
    scheduler = setup_scheduler(optimizer, args)
    
    train(
        model, 
        train_loader, 
        val_loader,
        optimizer,
        scheduler,
        device,
        args,
        cur_save_folder,
        local_rank
    )

if __name__ == "__main__":
    print("start")
    main()
