import os
import sys
import torch
import yaml
from model import VGG
from trainer import Trainer

from dataset import prepare_data
from transform import build_transforms



def load_params_from_yaml(file_path):
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def main():
    params = load_params_from_yaml('config/base.yaml')
    trainloader, validloader = prepare_data()
    model = VGG()
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=params['MODEL']['learning_rate'], 
        steps_per_epoch=len(trainloader), 
        epochs=params['MODEL']['epochs'],
        pct_start=0.2
    )
    trainer = Trainer(model, criterion, optimizer, scheduler)
    
    if os.path.exists(params['MODEL']['checkpoint_path']):
        print("[+] Found checkpoint. Loading model...")
        trainer.load_checkpoint(params['MODEL']['checkpoint_path'])

    try:
        trainer.fit(trainloader, validloader, epochs=params['MODEL']['epochs'])
    except KeyboardInterrupt:
        sys.exit()


if __name__ == "__main__":
    main()