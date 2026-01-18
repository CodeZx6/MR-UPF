import os
import argparse
import warnings
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

from config.registry import get_config
from data.loaders.dataloader import create_training_loaders, compute_model_parameters, load_semantic_descriptors
from data.preprocessors.normalization import MinMaxNormalizer
from utils.metrics import MetricAggregator, compute_rmse
from utils.initialization import InitializationStrategy
from core.networks.hierarchical_model import HierarchicalSpatioTemporalModel


class TrainingOrchestrator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = get_config()
        
        self.save_dir = self._create_save_directory()
        self.writer = self._create_tensorboard_writer()
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.normalizer = MinMaxNormalizer()
        
        self.best_rmse = np.inf
    
    def _create_save_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f'{self.args.dataset}_{self.args.base_channels}_{timestamp}'
        save_path = os.path.join('./checkpoints', dir_name)
        os.makedirs(save_path, exist_ok=True)
        return save_path
    
    def _create_tensorboard_writer(self):
        log_dir = os.path.join('./runs', os.path.basename(self.save_dir))
        return SummaryWriter(log_dir)
    
    def initialize_model(self):
        self.model = HierarchicalSpatioTemporalModel(
            in_channels=self.args.channels,
            out_channels=self.args.channels,
            img_height=self.args.img_height,
            img_width=self.args.img_width,
            base_channels=self.args.base_channels,
            beta_coeff=self.args.beta_coefficient
        )
        
        self.model = InitializationStrategy.apply(self.model)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        if self.device.type == 'cuda':
            self.model.cuda()
            self.criterion.cuda()
        
        compute_model_parameters(self.model, 'HierarchicalSTModel')
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2)
        )
    
    def load_pretrained_components(self):
        pass
    
    def train_epoch(self, train_loader, semantic_labels, epoch):
        self.model.train()
        epoch_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (temporal, context, text_tokens, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            predictions, semantic_loss = self.model(
                temporal, context, text_tokens,
                semantic_labels['binary'], semantic_labels['pentary'], semantic_labels['fine'],
                semantic_labels['binary'], semantic_labels['pentary'], semantic_labels['fine']
            )
            
            reconstruction_loss = self.criterion(predictions, targets)
            total_loss = reconstruction_loss / reconstruction_loss.detach() + semantic_loss / semantic_loss.detach()
            total_loss.backward()
            self.optimizer.step()
            
            predictions_denorm = self.normalizer.inverse_transform(predictions)
            targets_denorm = self.normalizer.inverse_transform(targets)
            batch_rmse = compute_rmse(predictions_denorm.cpu().detach().numpy(), 
                                     targets_denorm.cpu().detach().numpy())
            
            epoch_loss += reconstruction_loss.item() * len(targets)
            
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch}/{self.args.num_epochs}] [Batch {batch_idx}/{num_batches}] "
                      f"[RMSE: {batch_rmse:.6f}]")
        
        return epoch_loss / len(train_loader.dataset)
    
    def validate(self, val_loader, semantic_labels):
        self.model.eval()
        metrics = MetricAggregator()
        
        with torch.no_grad():
            for temporal, context, text_tokens, targets in val_loader:
                predictions, _ = self.model(
                    temporal, context, text_tokens,
                    semantic_labels['binary'], semantic_labels['pentary'], semantic_labels['fine'],
                    semantic_labels['binary'], semantic_labels['pentary'], semantic_labels['fine']
                )
                
                predictions = self.normalizer.inverse_transform(predictions).cpu().numpy()
                targets = self.normalizer.inverse_transform(targets).cpu().numpy()
                
                metrics.update(predictions, targets, len(targets))
        
        results = metrics.compute()
        return results
    
    def save_checkpoint(self, epoch, metrics):
        if metrics['rmse'] < self.best_rmse:
            self.best_rmse = metrics['rmse']
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            
            results_file = os.path.join(self.save_dir, 'results.txt')
            with open(results_file, 'a') as f:
                f.write(f"Epoch {epoch}: RMSE={metrics['rmse']:.6f}, "
                       f"MAE={metrics['mae']:.6f}, MAPE={metrics['mape']:.6f}\n")
            
            print(f"✓ New best model saved: RMSE={metrics['rmse']:.6f}")
    
    def adjust_learning_rate(self, epoch):
        if epoch > 0 and epoch % self.args.lr_decay_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Learning rate adjusted to {param_group['lr']}")
    
    def run(self):
        warnings.filterwarnings('ignore')
        
        print("Loading data...")
        train_loader, val_loader, test_loader, norm_stats = create_training_loaders(
            data_root=self.args.data_path,
            scale_x=self.args.scale_x,
            scale_y=self.args.scale_y,
            batch_size=self.args.batch_size
        )
        
        self.normalizer._max = norm_stats.item()['max']
        self.normalizer._min = norm_stats.item()['min']
        
        semantic_labels = load_semantic_descriptors(self.args.semantic_path)
        
        print("Initializing model...")
        self.initialize_model()
        self.load_pretrained_components()
        
        print("Starting training...")
        for epoch in range(self.args.num_epochs):
            epoch_start = datetime.now()
            
            self.adjust_learning_rate(epoch)
            
            train_loss = self.train_epoch(train_loader, semantic_labels, epoch)
            
            if (epoch + 1) % self.args.val_interval == 0:
                metrics = self.validate(val_loader, semantic_labels)
                print(f"Validation - RMSE: {metrics['rmse']:.6f}, "
                      f"MAE: {metrics['mae']:.6f}, MAPE: {metrics['mape']:.6f}")
                
                self.save_checkpoint(epoch, metrics)
                
                self.writer.add_scalar('metrics/rmse', metrics['rmse'], epoch)
                self.writer.add_scalar('metrics/mae', metrics['mae'], epoch)
                self.writer.add_scalar('loss/train', train_loss, epoch)
            
            print(f'Epoch {epoch} completed in {datetime.now() - epoch_start}')
        
        print("Evaluating final model...")
        final_metrics = self.validate(test_loader, semantic_labels)
        print(f"Final Test Results - RMSE: {final_metrics['rmse']:.6f}, "
              f"MAE: {final_metrics['mae']:.6f}, MAPE: {final_metrics['mape']:.6f}")
        
        self.writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Hierarchical Spatio-Temporal Learning Framework')
    
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--base_channels', type=int, default=128)
    parser.add_argument('--img_width', type=int, default=32)
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--val_interval', type=int, default=20)
    parser.add_argument('--lr_decay_epoch', type=int, default=50)
    parser.add_argument('--scale_x', type=int, default=1)
    parser.add_argument('--scale_y', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='TaxiBJ')
    parser.add_argument('--beta_coefficient', type=float, default=0.5)
    parser.add_argument('--data_path', type=str, default='./data/TaxiBJ')
    parser.add_argument('--semantic_path', type=str, default='./data')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    orchestrator = TrainingOrchestrator(args)
    orchestrator.run()
