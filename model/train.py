##################################################
# All functions related to training a model
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Michael Moeller
# Email: michael.moeller(at)uni-siegen.de
##################################################

import os
import random

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from misc.osutils import mkdir_if_missing
from misc.torchutils import count_parameters, seed_worker
from model.DeepConvLSTM import ConvBlock, ConvBlockSkip, ConvBlockFixup


def init_weights(network):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network.

    :param network: pytorch model
        Network of which weights are to be initialised
    :return: pytorch model
        Network with initialised weights
    """
    for m in network.modules():
        # normal convblock and skip convblock initialisation
        if isinstance(m, (ConvBlock, ConvBlockSkip)):
            if network.weights_init == 'normal':
                torch.nn.init.normal_(m.conv1.weight)
                torch.nn.init.normal_(m.conv2.weight)
            elif network.weights_init == 'orthogonal':
                torch.nn.init.orthogonal_(m.conv1.weight)
                torch.nn.init.orthogonal_(m.conv2.weight)
            elif network.weights_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.conv1.weight)
                torch.nn.init.xavier_uniform_(m.conv2.weight)
            elif network.weights_init == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.conv1.weight)
                torch.nn.init.xavier_normal_(m.conv2.weight)
            elif network.weights_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.conv1.weight)
                torch.nn.init.kaiming_uniform_(m.conv2.weight)
            elif network.weights_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.conv1.weight)
                torch.nn.init.kaiming_normal_(m.conv2.weight)
            m.conv1.bias.data.fill_(0.0)
            m.conv2.bias.data.fill_(0.0)
        # fixup block initialisation (see fixup paper for details)
        elif isinstance(m, ConvBlockFixup):
            nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * network.nb_conv_blocks ** (-0.5))
            nn.init.constant_(m.conv2.weight, 0)
        # linear layers
        elif isinstance(m, nn.Linear):
            if network.use_fixup:
                nn.init.constant_(m.weight, 0)
            elif network.weights_init == 'normal':
                torch.nn.init.normal_(m.weight)
            elif network.weights_init == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight)
            elif network.weights_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight)
            elif network.weights_init == 'xavier_normal':
                torch.nn.init.xavier_normal_(m.weight)
            elif network.weights_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight)
            elif network.weights_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # LSTM initialisation
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    if network.weights_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif network.weights_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif network.weights_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif network.weights_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif network.weights_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif network.weights_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    if network.weights_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif network.weights_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif network.weights_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif network.weights_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif network.weights_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif network.weights_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
    return network


def init_loss(config):
    """
    Initialises an loss object for a given network.

    :param config: dict
        General setting dictionary
    :return: loss object
    """
    if config.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(label_smoothing=config.smoothing)
    else:
        print("Did not provide a valid loss name!")
        return None
    return criterion


def init_optimizer(network, config):
    """
    Initialises an optimizer object for a given network.

    :param network: pytorch model
        Network for which optimizer and loss are to be initialised
    :param config: dict
        General setting dictionary
    :return: optimizer object
    """
    # define optimizer and loss
    if config.optimizer == 'adadelta':
        opt = torch.optim.Adadelta(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        opt = torch.optim.Adam(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        print("Did not provide a valid optimizer name!")
        return None
    return opt


def init_scheduler(optimizer, config):
    """

    :param optimizer: optimizer object
        Optimizer object used during training
    :param config: dict
        General setting dictionary
    :return:
    """
    if config.lr_scheduler == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_step, config.lr_decay)
    elif config.lr_scheduler == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.lr_step, factor=config.lr_decay)
    else:
        print("Did not provide a valid learning scheduler name!")
        return None
    return scheduler


def train(train_features, train_labels, val_features, val_labels, network, optimizer, loss, config, log_date, log_timestamp, lr_scheduler=None):
    """
    Method to train a PyTorch network.

    :param train_features: numpy array
        Training features
    :param train_labels: numpy array
        Training labels
    :param val_features: numpy array
        Validation features
    :param val_labels: numpy array
        Validation labels
    :param network: pytorch model
        DeepConvLSTM network object
    :param optimizer: optimizer object
        Optimizer object
    :param loss: loss object
        Loss object
    :param config: dict
        Config file which contains all training and hyperparameter settings
    :param log_date: string
        Date used for logging
    :param log_timestamp: string
        Timestamp used for logging
    :param lr_scheduler: scheduler object, default: None
        Learning rate scheduler object
    :return pytorch model, numpy array, numpy array
        Trained network and training and validation predictions with ground truth
    """
    log_dir = os.path.join('logs', log_date, log_timestamp)

    # prints the number of learnable parameters in the network
    count_parameters(network)

    # init network using weight initialization of choice
    network = init_weights(network)
    # send network to GPU
    network.to(config['gpu'])
    network.train()

    # if weighted loss chosen, calculate weights based on training dataset; else each class is weighted equally
    if config['weighted']:
        class_weights = torch.from_numpy(
            compute_class_weight('balanced', classes=np.unique(train_labels + 1), y=train_labels + 1)).float()
        if config['loss'] == 'cross_entropy':
            loss.weight = class_weights.cuda()
        print('Applied weighted class weights: ')
        print(class_weights)
    else:
        class_weights = torch.from_numpy(
            compute_class_weight(None, classes=np.unique(train_labels + 1), y=train_labels + 1)).float()
        if config['loss'] == 'cross_entropy':
            loss.weight = class_weights.cuda()

    # initialize optimizer and loss
    opt, criterion = optimizer, loss

    # initialize training and validation dataset, define DataLoaders
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))

    g = torch.Generator()
    g.manual_seed(config['seed'])

    trainloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=config['shuffling'],
                             worker_init_fn=seed_worker, generator=g, pin_memory=True)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
    valloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False,
                           worker_init_fn=seed_worker, generator=g, pin_memory=True)

    # counters and objects used for early stopping and learning rate adjustment
    best_metric = 0.0
    best_network = None
    best_val_losses = None
    best_train_losses = None
    best_val_preds = None
    best_train_preds = None
    early_stop = False
    es_pt_counter = 0

    # training loop; iterates through epochs
    for e in range(config['epochs']):
        """
        TRAINING
        """
        # helper objects
        train_preds = []
        train_gt = []
        train_losses = []
        start_time = time.time()
        batch_num = 1

        # iterate over train dataset
        for i, (x, y) in enumerate(trainloader):
            # send x and y to GPU
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])
            # zero accumulated gradients
            opt.zero_grad()

            if config['loss'] == 'maxup':
                # Increase the inputs via data augmentation
                inputs, targets = maxup(inputs, targets)

            # send inputs through network to get predictions, calculate loss and backpropagate
            train_output = network(inputs)

            train_loss = criterion(train_output, targets.long())

            train_loss.backward()
            opt.step()
            # append train loss to list
            train_losses.append(train_loss.item())

            # create predictions and append them to final list
            y_preds = np.argmax(train_output.cpu().detach().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            train_preds = np.concatenate((np.array(train_preds, int), np.array(y_preds, int)))
            train_gt = np.concatenate((np.array(train_gt, int), np.array(y_true, int)))

            # if verbose print out batch wise results (batch number, loss and time)
            if config['verbose']:
                if batch_num % config['print_freq'] == 0 and batch_num > 0:
                    cur_loss = np.mean(train_losses)
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d} batches | ms/batch {:5.2f} | '
                          'train loss {:5.2f}'.format(e, batch_num, elapsed * 1000 / config['batch_size'], cur_loss))
                    start_time = time.time()
                batch_num += 1

        """
        VALIDATION
        """

        # helper objects
        val_preds = []
        val_gt = []
        val_losses = []

        # set network to eval mode
        network.eval()
        with torch.no_grad():
            # iterate over validation dataset
            for i, (x, y) in enumerate(valloader):
                # send x and y to GPU
                inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

                # send inputs through network to get predictions, loss and calculate softmax probabilities
                val_output = network(inputs)

                val_loss = criterion(val_output, targets.long())

                val_output = torch.nn.functional.softmax(val_output, dim=1)

                # append validation loss to list
                val_losses.append(val_loss.item())

                # create predictions and append them to final list
                y_preds = np.argmax(val_output.cpu().numpy(), axis=-1)
                y_true = targets.cpu().numpy().flatten()
                val_preds = np.concatenate((np.array(val_preds, int), np.array(y_preds, int)))
                val_gt = np.concatenate((np.array(val_gt, int), np.array(y_true, int)))

            # print epoch evaluation results for train and validation dataset
            print("EPOCH: {}/{}".format(e + 1, config['epochs']),
                  "\nTrain Loss: {:.4f}".format(np.mean(train_losses)),
                  "Train Acc (M): {:.4f}".format(jaccard_score(train_gt, train_preds, average='macro')),
                  "Train Prc (M): {:.4f}".format(precision_score(train_gt, train_preds, average='macro')),
                  "Train Rcl (M): {:.4f}".format(recall_score(train_gt, train_preds, average='macro')),
                  "Train F1 (M): {:.4f}".format(f1_score(train_gt, train_preds, average='macro')),
                  "Train Acc (W): {:.4f}".format(jaccard_score(train_gt, train_preds, average='weighted')),
                  "Train Prc (W): {:.4f}".format(precision_score(train_gt, train_preds, average='weighted')),
                  "Train Rcl (W): {:.4f}".format(recall_score(train_gt, train_preds, average='weighted')),
                  "Train F1 (W): {:.4f}".format(f1_score(train_gt, train_preds, average='weighted')),
                  "\nValid Loss: {:.4f}".format(np.mean(val_losses)),
                  "Valid Acc (M): {:.4f}".format(jaccard_score(val_gt, val_preds, average='macro')),
                  "Valid Prc (M): {:.4f}".format(precision_score(val_gt, val_preds, average='macro')),
                  "Valid Rcl (M): {:.4f}".format(recall_score(val_gt, val_preds, average='macro')),
                  "Valid F1 (M): {:.4f}".format(f1_score(val_gt, val_preds, average='macro')),
                  "Valid Acc (W): {:.4f}".format(jaccard_score(val_gt, val_preds, average='weighted')),
                  "Valid Prc (W): {:.4f}".format(precision_score(val_gt, val_preds, average='weighted')),
                  "Valid Rcl (W): {:.4f}".format(recall_score(val_gt, val_preds, average='weighted')),
                  "Valid F1 (W): {:.4f}".format(f1_score(val_gt, val_preds, average='weighted'))
                  )

            # if chosen, print the value counts of the predicted labels for train and validation dataset
            if config['print_counts']:
                y_train = np.bincount(train_preds)
                ii_train = np.nonzero(y_train)[0]
                y_val = np.bincount(val_preds)
                ii_val = np.nonzero(y_val)[0]
                print('Predicted Train Labels: ')
                print(np.vstack((ii_train, y_train[ii_train])).T)
                print('Predicted Val Labels: ')
                print(np.vstack((ii_val, y_val[ii_val])).T)

        # adjust learning rate if enabled
        if config['adj_lr']:
            if config['lr_scheduler'] == 'reduce_lr_on_plateau':
                lr_scheduler.step(np.mean(val_losses))
            else:
                lr_scheduler.step()

        # employ early stopping if employed
        metric = f1_score(val_gt, val_preds, average='macro')
        if best_metric > metric:
            if config['early_stopping']:
                es_pt_counter += 1
                # early stopping check
                if es_pt_counter >= config['es_patience']:
                    print('Stopping training early since no loss improvement over {} epochs.'
                          .format(str(es_pt_counter)))
                    early_stop = True
                    # print results of best epoch
                    print('Final (best) results: ')
                    print("Train Loss: {:.4f}".format(np.mean(best_train_losses)),
                          "Train Acc: {:.4f}".format(jaccard_score(train_gt, best_train_preds, average='macro')),
                          "Train Prec: {:.4f}".format(precision_score(train_gt, best_train_preds, average='macro')),
                          "Train Rcll: {:.4f}".format(recall_score(train_gt, best_train_preds, average='macro')),
                          "Train F1: {:.4f}".format(f1_score(train_gt, best_train_preds, average='macro')),
                          "Val Loss: {:.4f}".format(np.mean(best_val_losses)),
                          "Val Acc: {:.4f}".format(jaccard_score(val_gt, best_val_preds, average='macro')),
                          "Val Prec: {:.4f}".format(precision_score(val_gt, best_val_preds, average='macro')),
                          "Val Rcll: {:.4f}".format(recall_score(val_gt, best_val_preds, average='macro')),
                          "Val F1: {:.4f}".format(f1_score(val_gt, best_val_preds, average='macro')))
        else:
            print(f"Performance improved... ({best_metric}->{metric})")
            if config['early_stopping']:
                es_pt_counter = 0
                best_train_losses = train_losses
                best_val_losses = val_losses
            best_metric = metric
            best_network = network
            checkpoint = {
                "model_state_dict": network.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "random_rnd_state": random.getstate(),
                "numpy_rnd_state": np.random.get_state(),
                "torch_rnd_state": torch.get_rng_state(),
            }
            best_train_preds = train_preds
            best_val_preds = val_preds

        # set network to train mode again
        network.train()

        if early_stop:
            break

    # return validation, train and test predictions as numpy array with ground truth
    if config['valid_epoch'] == 'best':
        return best_network, checkpoint, np.vstack((best_val_preds, val_gt)).T, \
               np.vstack((best_train_preds, train_gt)).T
    else:
        checkpoint = {
            "model_state_dict": network.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }
        return network, checkpoint, np.vstack((val_preds, val_gt)).T, np.vstack((train_preds, train_gt)).T


def predict(test_features, test_labels, network, config, log_date, log_timestamp):
    """
    Method that applies a trained network to obtain predictions on a test dataset. If selected, saves predictions.

    :param test_features: numpy array
        Test features
    :param test_labels: numpy array
        Test labels
    :param network: pytorch model
        Trained network object
    :param config: dict
        Config file which contains all training and hyperparameter settings
    :param log_date: string
        Date used for saving predictions
    :param log_timestamp: string
        Timestamp used for saving predictions
    """
    log_dir = os.path.join('logs', log_date, log_timestamp)

    # set network to eval mode
    network.eval()
    # helper objects
    test_preds = []
    test_gt = []

    # initialize test dataset and loader
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_features).float(), torch.from_numpy(test_labels))
    testloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            worker_init_fn=np.random.seed(int(config['seed']))
                            )

    with torch.no_grad():
        # iterate over test dataset
        for i, (x, y) in enumerate(testloader):
            # send x and y to GPU
            inputs, targets = x.to(config['gpu']), y.to(config['gpu'])

            # send inputs through network to get predictions and calculate softmax probabilities
            test_output = network(inputs)
            test_output = torch.nn.functional.softmax(test_output, dim=1)

            # create predictions and append them to final list
            y_preds = np.argmax(test_output.cpu().numpy(), axis=-1)
            y_true = targets.cpu().numpy().flatten()
            test_preds = np.concatenate((np.array(test_preds, int), np.array(y_preds, int)))
            test_gt = np.concatenate((np.array(test_gt, int), np.array(y_true, int)))

    print('\nTEST RESULTS: ')
    print("Avg. Accuracy: {0}".format(jaccard_score(test_gt, test_preds, average='macro')))
    print("Avg. Precision: {0}".format(precision_score(test_gt, test_preds, average='macro')))
    print("Avg. Recall: {0}".format(recall_score(test_gt, test_preds, average='macro')))
    print("Avg. F1: {0}".format(f1_score(test_gt, test_preds, average='macro')))

    print("\nTEST RESULTS (PER CLASS): ")
    print("Accuracy: {0}".format(jaccard_score(test_gt, test_preds, average=None)))
    print("Precision: {0}".format(precision_score(test_gt, test_preds, average=None)))
    print("Recall: {0}".format(recall_score(test_gt, test_preds, average=None)))
    print("F1: {0}".format(f1_score(test_gt, test_preds, average=None)))

    if config['save_test_preds']:
        mkdir_if_missing(log_dir)
        if config['name']:
            np.save(os.path.join(log_dir, 'test_preds_{}.npy'.format(config['name'])), test_output.cpu().numpy())
        else:
            np.save(os.path.join(log_dir, 'test_preds.npy'), test_output.cpu().numpy())
