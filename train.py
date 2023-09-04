import torch.nn as nn
import torch
import os
import random
import argparse
import numpy as np
from utils import generate_loaders, generate_datasets
from model import MWTCNN

num_epochs = 60
batch_size = 16
learning_rate = 0.001
weight_decay = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_weights(model):
    """ Use He initialization for the model weights. """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def train(train_loader, val_loader, model, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        # if epoch >= 29:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass, get the loss values
            images = images.to(device).view(batch_size, 1, 128, 128)
            labels = labels.long().to(device)
            labels_predicted = model(images)
            loss = criterion(labels_predicted, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)  # Calculate average training loss
        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device).view(batch_size, 1, 128, 128)
                labels = labels.long().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Calculate average validation loss
        val_loss /= len(val_loader)
        # Print epoch details
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # scheduler.step()
        model.train()
    return model


def test(model_trained, test_loader):
    model_trained.train()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device).view(batch_size, 1, 128, 128)
            labels = labels.long().to(device)
            outputs = model_trained(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))


def train_all(train_from_scratch=True, pre_model_path=None):
    ''' Prepare data '''
    img_path = './data/grayscale_images/gray_images_all.npy'
    label_path = './data/npy_files/labels_uwb_dataset_all.npy'
    print("Training on the part 1-7 dataset...")
    train_dataset, val_dataset, test_dataset = generate_datasets(image_array_path=img_path,
                                                                 label_array_path=label_path)
    train_loader, val_loader, test_loader = generate_loaders(train_dataset, val_dataset, test_dataset,
                                                             batch_size=batch_size)
    model = MWTCNN()
    if not train_from_scratch:
        model.load_state_dict(torch.load(pre_model_path))
    else:
        initialize_weights(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True,
                                weight_decay=weight_decay)
    model = train(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion,
                  optimizer=optimizer)
    model_path = './models/MWT_CNN_v2_1'
    torch.save(model.state_dict(), model_path)
    test(model_trained=model, test_loader=test_loader)


def train_part1(train_from_scratch=True, pre_model_path=None):
    ''' Prepare data '''
    img_path = './data/grayscale_images/gray_images_part1.npy'
    label_path = './data/npy_files/labels_uwb_dataset_part1.npy'
    print("Training on the part1-dataset...")
    train_dataset, val_dataset, test_dataset = generate_datasets(image_array_path=img_path,
                                                                 label_array_path=label_path)
    train_loader, val_loader, test_loader = generate_loaders(train_dataset, val_dataset, test_dataset, batch_size=16)
    model = MWTCNN()
    if not train_from_scratch:
        model.load_state_dict(torch.load(pre_model_path))
    else:
        initialize_weights(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    model = train(train_loader=train_loader, val_loader=val_loader, model=model, criterion=criterion,
                  optimizer=optimizer)
    model_path = './models/MWT_CNN_part1'
    torch.save(model.state_dict(), model_path)
    test(model_trained=model, test_loader=test_loader)


if __name__ == '__main__':
    ''' Train on the part1 dataset '''
    # train_from_scratch = False
    train_from_scratch = True
    # pre_model_path = './models/MWT_CNN_part1'
    pre_model_path = './models/MWT_CNN_v2_1'
    # train_part1(train_from_scratch=train_from_scratch, pre_model_path=pre_model_path)
    train_all(train_from_scratch=train_from_scratch, pre_model_path=pre_model_path)
