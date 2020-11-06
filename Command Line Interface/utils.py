import argparse
import json
import torch
from torchvision import transforms, datasets, models

def load_category_names(filepath):
    with open (filepath) as file:
        cat_to_name = json.load(file)
    return cat_to_name
    
def save_model_checkpoint(path, model, optimizer, classifier, args):
    checkpoint = {'model': model,
                  'classifier': classifier,
                  'epochs': args.epochs,
                  'arch': args.arch,
                  'lr': args.learnrate,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
                  
    torch.save(checkpoint, path)
    
def load_model_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    learnrate = checkpoint['lr']
    arch = checkpoint['arch']
    optimizer = checkpoint['optimizer']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

