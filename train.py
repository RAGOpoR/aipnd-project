import torch 
import torch.nn as tnn
import torch.optim as toptim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import pandas
import matplotlib.pyplot as plt
import time
import json
from collections import OrderedDict
from torchvision import datasets, models, transforms, utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import argparse

use_gpu = torch.cuda.is_available()

def initial_data(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
    }
    
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    batch_size = 8
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size)
    dataloaders['test']  = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)
    
    return dataloaders, image_datasets

def train_model_manager(args, model, criterion, optimizer, scheduler, num_epochs=10):
  
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    dataloaders, image_datasets = initial_data(args)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    
    for epoch in range(num_epochs):
        print('Epoch {} / {}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_accuracy))

            if phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training has complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy : {:4f}'.format(best_accuracy))

    model.load_state_dict(best_model_wts)
    return model

def save_check_point(checkpoint):
    torch.save(checkpoint, 'current_checkpoint.pth')

def setup_train_model(args):
    dataloaders, image_datasets = initial_data(args)
    
    if args.arch == 'vgg': 
        model = models.vgg16(pretrained=True)
    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.classifier[0].in_features
    classifier = tnn.Sequential(OrderedDict([
                                  ('fc1', tnn.Linear(num_features, 512)),
                                  ('relu', tnn.ReLU()),
                                  ('drpot', tnn.Dropout(p=0.5)),
                                  ('hidden', tnn.Linear(512, args.hidden_units)),                       
                                  ('fc2', tnn.Linear(args.hidden_units, 102)),
                                  ('output', tnn.LogSoftmax(dim=1)),
                                  ]))

    model.classifier = classifier
    
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU : "+ str(use_gpu))
        else:
            print("GPU is not available")
            
  
    num_epochs = 10

    criterion = tnn.CrossEntropyLoss()
    optimizer = toptim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model_manager(args, model, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs)
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.epochs = num_epochs
    checkpoint = {'input_size': [3, 224, 224],
                     'batch_size': dataloaders['train'].batch_size,
                      'output_size': 102,
                      'state_dict': model.state_dict(),
                      'optimizer_dict':optimizer.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'epoch': model.epochs}
    save_check_point(checkpoint)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=False, help='Use GPU if available')
    parser.add_argument('--data_dir', type=str, help='Path to dataset ')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--arch', type=str, help='Model architecture')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
    args = parser.parse_args()
    
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)

    setup_train_model(args)


if __name__ == "__main__":
    main()