##################################################
# All functions related to validating a model
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
##################################################

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data_processing.sliding_window import apply_sliding_window
from misc.osutils import mkdir_if_missing
from model.DeepConvLSTM import DeepConvLSTM
from model.evaluate import evaluate_participant_scores
from model.train import train, init_optimizer, init_loss, init_scheduler


def cross_participant_cv(data, custom_net, custom_loss, custom_opt, args, log_date, log_timestamp):
    """
    Method to apply cross-participant cross-validation (also known as leave-one-subject-out cross-validation).

    :param data: numpy array
        Data used for applying cross-validation
    :param custom_net: pytorch model
        Custom network object
    :param custom_loss: loss object
        Custom loss object
    :param custom_opt: optimizer object
        Custom optimizer object
    :param args: dict
        Args object containing all relevant hyperparameters and settings
    :param log_date: string
        Date information needed for saving
    :param log_timestamp: string
        Timestamp information needed for saving
    :return pytorch model
        Trained network
    """

    print('\nCALCULATING CROSS-PARTICIPANT SCORES USING LOSO CV.\n')
    cp_scores = np.zeros((4, args.nb_classes, int(np.max(data[:, 0]) + 1)))
    train_val_gap = np.zeros((4, int(np.max(data[:, 0]) + 1)))
    all_eval_output = None
    orig_lr = args.learning_rate
    log_dir = os.path.join('logs', log_date, log_timestamp)

    for i, sbj in enumerate(np.unique(data[:, 0])):
        # for i, sbj in enumerate([0, 1]):
        print('\n VALIDATING FOR SUBJECT {0} OF {1}'.format(int(sbj) + 1, int(np.max(data[:, 0])) + 1))
        train_data = data[data[:, 0] != sbj]
        val_data = data[data[:, 0] == sbj]
        args.learning_rate = orig_lr
        # Sensor data is segmented using a sliding window mechanism
        X_train, y_train = apply_sliding_window(train_data[:, :-1], train_data[:, -1],
                                                sliding_window_size=args.sw_length,
                                                unit=args.sw_unit,
                                                sampling_rate=args.sampling_rate,
                                                sliding_window_overlap=args.sw_overlap,
                                                )

        X_val, y_val = apply_sliding_window(val_data[:, :-1], val_data[:, -1],
                                            sliding_window_size=args.sw_length,
                                            unit=args.sw_unit,
                                            sampling_rate=args.sampling_rate,
                                            sliding_window_overlap=args.sw_overlap,
                                            )

        X_train, X_val = X_train[:, :, 1:], X_val[:, :, 1:]

        args.window_size = X_train.shape[1]
        args.nb_channels = X_train.shape[2]

        # network initialization
        if args.network == 'deepconvlstm':
            net = DeepConvLSTM(config=vars(args))
        elif args.network == 'custom':
            net = custom_net
        else:
            print("Did not provide a valid network name!")

        # optimizer initialization
        if args.optimizer != 'custom':
            opt = init_optimizer(net, args)
        else:
            opt = custom_opt

        # optimizer initialization
        if args.loss != 'custom':
            loss = init_loss(args)
        else:
            loss = custom_loss

        # lr scheduler initialization
        if args.adj_lr:
            print('Adjusting learning rate according to scheduler: ' + args.lr_scheduler)
            scheduler = init_scheduler(opt, args)
        else:
            scheduler = None

        net, checkpoint, val_output, train_output = train(X_train, y_train, X_val, y_val,
                                                          network=net, optimizer=opt, loss=loss, lr_scheduler=scheduler,
                                                          config=vars(args), log_date=log_date,
                                                          log_timestamp=log_timestamp)

        if args.save_checkpoints:
            mkdir_if_missing(log_dir)
            print('Saving checkpoint...')
            if args.valid_epoch == 'last':
                if args.name:
                    c_name = os.path.join(log_dir, "checkpoint_last_{}_{}.pth".format(str(sbj), str(args.name)))
                else:
                    c_name = os.path.join(log_dir, "checkpoint_last_{}.pth".format(str(sbj)))
            else:
                if args.name:
                    c_name = os.path.join(log_dir, "checkpoint_best_{}_{}.pth".format(str(sbj), str(args.name)))
                else:
                    c_name = os.path.join(log_dir, "checkpoint_best_{}.pth".format(str(sbj)))
            torch.save(checkpoint, c_name)

        if all_eval_output is None:
            all_eval_output = val_output
        else:
            all_eval_output = np.concatenate((all_eval_output, val_output), axis=0)

        # fill values for normal evaluation
        labels = list(range(0, args.nb_classes))
        cp_scores[0, :, int(sbj)] = jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[1, :, int(sbj)] = precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[2, :, int(sbj)] = recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)
        cp_scores[3, :, int(sbj)] = f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)

        # fill values for train val gap evaluation
        train_val_gap[0, int(sbj)] = jaccard_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     jaccard_score(val_output[:, 1], val_output[:, 0], average='macro', labels=labels)
        train_val_gap[1, int(sbj)] = precision_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     precision_score(val_output[:, 1], val_output[:, 0], average='macro', labels=labels)
        train_val_gap[2, int(sbj)] = recall_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     recall_score(val_output[:, 1], val_output[:, 0], average='macro', labels=labels)
        train_val_gap[3, int(sbj)] = f1_score(train_output[:, 1], train_output[:, 0], average='macro', labels=labels) - \
                                     f1_score(val_output[:, 1], val_output[:, 0], average='macro', labels=labels)

        print("SUBJECT {0} VALIDATION RESULTS: ".format(int(sbj) + 1))
        print("Accuracy: {0}".format(jaccard_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)))
        print("Precision: {0}".format(precision_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)))
        print("Recall: {0}".format(recall_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)))
        print("F1: {0}".format(f1_score(val_output[:, 1], val_output[:, 0], average=None, labels=labels)))

    if args.save_analysis:
        mkdir_if_missing(log_dir)
        cp_score_acc = pd.DataFrame(cp_scores[0, :, :], index=None)
        cp_score_acc.index = args.class_names
        cp_score_prec = pd.DataFrame(cp_scores[1, :, :], index=None)
        cp_score_prec.index = args.class_names
        cp_score_rec = pd.DataFrame(cp_scores[2, :, :], index=None)
        cp_score_rec.index = args.class_names
        cp_score_f1 = pd.DataFrame(cp_scores[3, :, :], index=None)
        cp_score_f1.index = args.class_names
        tv_gap = pd.DataFrame(train_val_gap, index=None)
        tv_gap.index = ['accuracy', 'precision', 'recall', 'f1']
        if args.name:
            cp_score_acc.to_csv(os.path.join(log_dir, 'cp_scores_acc_{}.csv'.format(args.name)))
            cp_score_prec.to_csv(os.path.join(log_dir, 'cp_scores_prec_{}.csv').format(args.name))
            cp_score_rec.to_csv(os.path.join(log_dir, 'cp_scores_rec_{}.csv').format(args.name))
            cp_score_f1.to_csv(os.path.join(log_dir, 'cp_scores_f1_{}.csv').format(args.name))
            tv_gap.to_csv(os.path.join(log_dir, 'train_val_gap_{}.csv').format(args.name))
        else:
            cp_score_acc.to_csv(os.path.join(log_dir, 'cp_scores_acc.csv'))
            cp_score_prec.to_csv(os.path.join(log_dir, 'cp_scores_prec.csv'))
            cp_score_rec.to_csv(os.path.join(log_dir, 'cp_scores_rec.csv'))
            cp_score_f1.to_csv(os.path.join(log_dir, 'cp_scores_f1.csv'))
            tv_gap.to_csv(os.path.join(log_dir, 'train_val_gap.csv'))

    evaluate_participant_scores(participant_scores=cp_scores,
                                gen_gap_scores=train_val_gap,
                                input_cm=all_eval_output,
                                class_names=args.class_names,
                                nb_subjects=int(np.max(data[:, 0]) + 1),
                                filepath=os.path.join('logs', log_date, log_timestamp),
                                filename='cross-participant',
                                args=args
                                )

    return net
