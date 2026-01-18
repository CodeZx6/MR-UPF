import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def compute_model_parameters(model, name='Model'):
    total_params = sum([p.nelement() for p in model.parameters()])
    print(f'{name} parameters: {total_params}')
    return total_params


class DataLoaderFactory:
    def __init__(self, data_root, scale_x=1, scale_y=1, batch_size=128):
        self.data_root = data_root
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _load_arrays(self, mode):
        mode_path = os.path.join(self.data_root, mode)
        
        arrays = {}
        arrays['target'] = np.load(os.path.join(mode_path, 'basis.npy'))
        arrays['temporal'] = np.load(os.path.join(mode_path, 'time_correlation.npy'))
        arrays['context'] = np.load(os.path.join(mode_path, 'time_c_feature.npy'))
        arrays['text_tokens'] = np.load(os.path.join(mode_path, 'time_text.npy'))
        
        return arrays
    
    def _prepare_tensors(self, arrays):
        Tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        
        temporal = Tensor(arrays['temporal']) / self.scale_x
        target = Tensor(arrays['target']) / self.scale_y
        context = Tensor(arrays['context'].astype(np.float32))
        text_tokens = torch.LongTensor(arrays['text_tokens'])
        
        return temporal, context, text_tokens, target
    
    def create_loader(self, mode='train', shuffle=None):
        arrays = self._load_arrays(mode)
        tensors = self._prepare_tensors(arrays)
        
        dataset = TensorDataset(*tensors)
        
        if shuffle is None:
            shuffle = (mode == 'train')
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        print(f'# {mode} samples: {len(dataset)}')
        return loader
    
    @staticmethod
    def load_normalization_stats(data_root):
        return np.load(os.path.join(data_root, 'basis.npy'), allow_pickle=True)


def load_semantic_descriptors(descriptor_path):
    spatial_descriptors = {}
    labels = {}
    
    for scale, filename in [('binary', '2_data.npy'), 
                            ('pentary', '5_data.npy'), 
                            ('fine', 'n_data.npy')]:
        data = np.load(os.path.join(descriptor_path, filename), allow_pickle=True).item()
        labels[scale] = data['label']
    
    return labels


def create_training_loaders(data_root, scale_x, scale_y, batch_size):
    factory = DataLoaderFactory(data_root, scale_x, scale_y, batch_size)
    
    train_loader = factory.create_loader('train', shuffle=True)
    val_loader = factory.create_loader('test', shuffle=False)
    test_loader = factory.create_loader('test', shuffle=False)
    norm_stats = factory.load_normalization_stats(data_root)
    
    return train_loader, val_loader, test_loader, norm_stats
