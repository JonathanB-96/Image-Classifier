import argparse

import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch import nn, optim

import numpy as py
import matplotlib.pyplot as plt

from PIL import Image

import utils

def get_input_args():
    
    parser = argparse.ArgumentParser(description = "Train")
    
    parser.add_argument('--data_dir')
    parser.add_argument('--arch', dest = 'arch', default = 'alexnet', choices=['alexnet', 'resnet50', 'vgg16'], help = 'The CNN architecture')
    parser.add_argument('--learnrate', dest = 'learnrate', default = '0.001')
    parser.add_argument('--epochs', dest = 'epochs', default = '5')
    parser.add_argument('--gpu', default = 'gpu')
    parser.add_argument('-hidden_layers', type = int, dest = 'hidden_layers', default = '512', help = 'Set the number of hidden layers')
    parser.add_argument('--save_dir', dest = 'save_dir', default = 'ImageClassifier/checkpoint.pth', help = 'Path to saved model checkpoint')
    
    return parser.parse_args()

def train(epochs, trainloader, device, optimizer, model, criterion, validloader):
    
    print_every = 24
    steps = 0
    
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            
            if device == 'gpu':
                model.to('cuda')
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                model.to('cpu')
            
            optimizer.zero_grad()
            
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for images, labels in validloader:
                        
                        if device == 'gpu':
                            model.to('cuda')
                            images, labels = images.to('cuda'), labels.to('cuda')
                        else:
                            model.to('cpu')
                        
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                            
                running_loss = 0
                model.train()

def main():
    
    args = get_input_args()
    
    data_dir = 'ImageClassifier/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = test_valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)
    
    model = getattr(models, args.arch)(pretrained=True)
    
    for param in model.parameters():
        param.required_grad = False
    
    if args.arch == 'alexnet':
        
        classifier = nn.Sequential(nn.Linear(9216, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(4096,1000),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(1000,102),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier

    elif args.arch == 'vgg16':
        
        classifier = nn.Sequential(nn.Linear(25088, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(4096,1000),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(1000,102),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier
        
    elif args.arch == 'resnet50':
        
        classifier = nn.Sequential(nn.Linear(2048, 512),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(512,102),
                                   nn.LogSoftmax(dim=1))
        model.fc = classifier
    
    criterion = nn.NLLLoss()
    if args.arch == 'resnet50':
        optimizer = optim.Adam(model.fc.parameters(), lr = float(args.learnrate))
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr = float(args.learnrate))
    epochs = int(args.epochs)
    class_idx = train_dataset.class_to_idx
    device = args.gpu
    
    train(epochs, trainloader, device, optimizer, model, criterion, validloader)
    
    model.class_to_idx = class_idx
    path = args.save_dir
    
    utils.save_model_checkpoint(path, model, optimizer, classifier, args)
    
main()
    