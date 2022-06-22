import torch

config = {
    'default_device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'default_classes': [
     'live_fish', 'morts'
    ]
}
