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
from PIL import Image

use_gpu = torch.cuda.is_available

def image_processing(image):
    
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npimage = np.array(image)
    npimage = npimage/255.0
    imga = npimage[:,:,0]
    imgb = npimage[:,:,1]
    imgc = npimage[:,:,2]
    imga = (imga - 0.485)/(0.229) 
    imgb = (imgb - 0.456)/(0.224)
    imgc = (imgc - 0.406)/(0.225)
    npimage[:,:,0] = imga
    npimage[:,:,1] = imgb
    npimage[:,:,2] = imgc
    npimage = np.transpose(npimage, (2,0,1))
    return npimage

def predict_manager(args, image_path, model, class_to_idx, idx_to_class, cat_to_name, topk=5):
    
    image = torch.FloatTensor([image_processing(Image.open(image_path))])
    if args.gpu and use_gpu:
        model = model.cuda()    
    model.eval()

    if args.gpu and use_gpu:
        output = model.forward(Variable(image.cuda()))
    else:
        output = model.forward(Variable(image))
    
    probability = torch.exp(output.cpu()).data.numpy()[0]

    top_idx = np.argsort(probability)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = probability[top_idx]

    return top_probability, top_class

def restore_checkpoint(args):
    
    checkpoint = torch.load(args.saved_model)
    if args.arch == 'vgg':
        model = models.vgg16()        
    elif args.arch == 'densenet':
        model = models.densenet121()

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
    model.load_state_dict(checkpoint['state_dict'])
    
    if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU")
        else:
            print("GPU is not available")

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = { v : k for k,v in class_to_idx.items()}
    return model, class_to_idx, idx_to_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=False, help='Use GPU if available')
    parser.add_argument('--image', type=str, help='Image for predict')
    parser.add_argument('--hidden_units', type=int, default=100, help='hidden units for fc layer')
    parser.add_argument('--saved_model' , type=str, default='current_checkpoint.pth', help='path of saved model')
    parser.add_argument('--mapper_json' , type=str, default='cat_to_name.json', help='path of your mapper from category to name')
    parser.add_argument('--topk', type=int, default=5, help='display top k prob')
    parser.add_argument('--arch', type=str, help='Model architecture')
    args = parser.parse_args()
    
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)

    model, class_to_idx, idx_to_class = restore_checkpoint(args)
    top_probability, top_class = predict_manager(args, args.image, model, class_to_idx, idx_to_class, cat_to_name, topk=args.topk)
                                              
    print('Predicted Classes: ', top_class)
    print ('Class Names: ')
    [print(cat_to_name[x]) for x in top_class]
    print('Predicted Probability: ', top_probability)


if __name__ == "__main__":
    main()