import argparse

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import json

from PIL import Image

import utils

def get_input_args():
    
    parser = argparse.ArgumentParser(description = "Predict")
    
    parser.add_argument('--checkpoint', default = 'ImageClassifier/checkpoint.pth')
    parser.add_argument('--topk', type = int, dest = 'topk', default = '5', help = 'Set the value for Top K classes')
    parser.add_argument('--path_to_image', dest = 'path_to_image', default = 'ImageClassifier/flowers/test/100/image_07896.jpg', help = 'Path to Image')
    parser.add_argument('--categories', dest = 'categories', default = 'ImageClassifier/cat_to_name.json', help = 'JSON file to get category names')
    parser.add_argument('--device', default = 'gpu')
    
    return parser.parse_args()

def process_image(filepath):
    
    image = Image.open(filepath)
    
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0,406],
                                                                [0.229, 0.224, 0.225])])
    image = image_transforms(image)
    return image

def predict(path_to_image, model, topk, device):
    
    if device == 'gpu': 
        model.to('cuda')
    else:
        model.to('cpu')
       
    model.eval()
    
    image = process_image(path_to_image)
    image = image.numpy()
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0).float()
    
    if device == 'gpu':
        image = image.to('cuda')
            
    else:
        image = image.to('cpu')
        
    with torch.no_grad():
        logps = model.forward(image)
        
    ps = torch.exp(logps)
    top_ps = np.array(ps.topk(topk, dim=1)[0][0])
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [np.int(idx_to_class[each]) for each in np.array(ps.topk(topk, dim=1)[1][0])]
                                                                 
    return top_ps, top_class
                                                                 
def main():
    args = get_input_args()
    device = args.device
    topk = args.topk
    imagepath = args.path_to_image                                                             
    categories = args.categories                                                             
    checkpoint = args.checkpoint
    
    model = utils.load_model_checkpoint(checkpoint)
    cat_to_name = utils.load_category_names(categories)
                                                                 
    top_ps, top_class = predict(imagepath, model, topk, device)
    category = [cat_to_name[str(index)] for index in top_class]                                                                 
    
    for i in range(len(category)):
        print("Category {} has probablity {}".format(category[i], top_ps[i]))                                                                 
                                                                 
main()                                                                