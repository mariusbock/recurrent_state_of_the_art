##################################################
# All functions related to preprocessing and loading data
##################################################
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# Author: Alexander Hölzemann
# Email: alexander.hoelzemann(at)uni-siegen.de
##################################################

import os

import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


def normalize(df):
    """
    Based on: https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = 2 * (df[feature_name] - min_value) / (max_value - min_value) - 1
    return result


def load_dataset(dataset, pred_type='actions', include_null=False):
    """
    Main function to load one of the supported datasets

    :param dataset: string
        Name of dataset to be loaded
    :param pred_type: string
        Prediction type which is to be used (if multi-target dataset)
    :param include_null: boolean, default: False
        Whether to include null class in dataframe
    :return: numpy float arrays, int, list of strings, int, boolean
        features, labels, number of classes, class names, sampling rate and boolean has_null
    """
    if dataset == 'wetlab':
        if pred_type == 'actions':
            class_names = ['cutting', 'inverting', 'peeling', 'pestling', 'pipetting', 'pouring', 'stirring',
                           'transfer']
        elif pred_type == 'tasks':
            class_names = ['1solvent', '2catalysator', '3cutting', '4mixing', '5catalysator', '6waterbath', '7solvent',
                           '8catalysator', '9cutting', '10mixing', '11catalysator', '12waterbath', '13waterbath',
                           '14catalysator', '15pestling', '16filtrate', '17catalysator', '18pouring', '19detect',
                           '20waterbath', '21catalysator', '22pestling', '23filtrate', '24catalysator', '25pouring',
                           '26detect', '27end']
        sampling_rate = 50
        has_null = True
    elif dataset == 'sbhar':
        class_names = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'laying',
                       'stand_to_sit', 'sit_to_stand', 'sit_to_lie', 'lie_to_sit', 'stand_to_lie', 'lie_to_stand']
        sampling_rate = 50
        has_null = True
    elif dataset == 'rwhar':
        class_names = ['climbing_down', 'climbing_up', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
        sampling_rate = 50
        has_null = False
    elif dataset == 'hhar':
        class_names = ['biking', 'sitting', 'standing', 'walking', 'stair up', 'stair down']
        sampling_rate = 100
        has_null = True
    elif dataset == 'opportunity' or dataset == 'opportunity_full' or dataset == 'opportunity_adl' or dataset == 'opportunity_drill':
        sampling_rate = 30
        has_null = True
        if pred_type == 'gestures':
            class_names = ['open_door_1', 'open_door_2', 'close_door_1', 'close_door_2', 'open_fridge',
                           'close_fridge', 'open_dishwasher', 'close_dishwasher', 'open_drawer_1', 'close_drawer_1',
                           'open_drawer_2', 'close_drawer_2', 'open_drawer_3', 'close_drawer_3', 'clean_table',
                           'drink_from_cup', 'toggle_switch']
        elif pred_type == 'locomotion':
            class_names = ['stand', 'walk', 'sit', 'lie']

    data = pd.read_csv(os.path.join('data/', dataset + '_data.csv'), sep=',', header=None, index_col=None)

    X, y = preprocess_data(data, dataset, pred_type, has_null, include_null)

    print(" ..from file {}".format(os.path.join('data/', dataset + '_data.csv')))
    X = X.astype(np.float32)
    y = y.astype(np.uint8)

    if has_null and include_null:
        class_names = ['null'] + class_names

    return X, y, len(class_names), class_names, sampling_rate, has_null


def preprocess_data(data, ds, pt='actions', has_null=False, include_null=True):
    """
    Function to preprocess the wetlab dataset according to settings.
    :param data: pandas dataframe
        Dataframe containing all data
    :param ds: string
        Name of dataset
    :param pt: string, ['actions' (default), 'tasks']
        Type of labels that are to be used
    :param has_null: boolean, default: False
        Boolean signaling whether dataset has a null class
    :param include_null: boolean, default: True
        Boolean signaling whether to include or not include the null class in the dataset
    :return numpy float arrays
        Training and validation datasets that can be used for training
    """

    print('Processing dataset files ...')
    if has_null:
        if include_null:
            pass
        else:
            if (ds == 'wetlab' and pt == 'actions') or ((ds == 'opportunity' or ds == 'opportunity_full' or ds == 'opportunity_adl' or ds == 'opportunity_drill') and pt == 'locomotion'):
                data = data[(data.iloc[:, -2] != 'null_class')]
            else:
                data = data[(data.iloc[:, -1] != 'null_class')]

    if (ds == 'wetlab' and pt == 'actions') or ((ds == 'opportunity' or ds == 'opportunity_full' or ds == 'opportunity_adl' or ds == 'opportunity_drill') and pt == "locomotion"):
        X, y = data.iloc[:, :-2], adjust_labels(data.iloc[:, -2], ds, pt).astype(int)
    elif (ds == 'wetlab' and pt == 'tasks') or ((ds == 'opportunity' or ds == 'opportunity_full' or ds == 'opportunity_adl' or ds == 'opportunity_drill') and pt == "gestures"):
        X, y = data.iloc[:, :-2], adjust_labels(data.iloc[:, -1], ds, pt).astype(int)
    else:
        X, y = data.iloc[:, :-1], adjust_labels(data.iloc[:, -1], ds, pt).astype(int)

    # if no null class in dataset subtract one from all labels
    if has_null and not include_null:
        y -= 1

    print("Full dataset with size: | X {0} | y {1} | ".format(X.shape, y.shape))

    return X, y


def adjust_labels(data_y, dataset, pred_type='actions'):
    """
    Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param dataset: string
        String indicating which dataset is to be adjusted
    :param pred_type: string, ['gestures', 'locomotion', 'actions', 'tasks']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """
    data_y[data_y == "null_class"] = 0
    if dataset == 'wetlab':
        if pred_type == 'tasks':  # Labels for tasks are adjusted
            data_y[data_y == "1solvent"] = 1
            data_y[data_y == "2catalysator"] = 2
            data_y[data_y == "3cutting"] = 3
            data_y[data_y == "4mixing"] = 4
            data_y[data_y == "5catalysator"] = 5
            data_y[data_y == "6waterbath"] = 6
            data_y[data_y == "7solvent"] = 7
            data_y[data_y == "8catalysator"] = 8
            data_y[data_y == "9cutting"] = 9
            data_y[data_y == "10mixing"] = 10
            data_y[data_y == "11catalysator"] = 11
            data_y[data_y == "12waterbath"] = 12
            data_y[data_y == "13waterbath"] = 13
            data_y[data_y == "14catalysator"] = 14
            data_y[data_y == "15pestling"] = 15
            data_y[data_y == "16filtrate"] = 16
            data_y[data_y == "17catalysator"] = 17
            data_y[data_y == "18pouring"] = 18
            data_y[data_y == "19detect"] = 19
            data_y[data_y == "20waterbath"] = 20
            data_y[data_y == "21catalysator"] = 21
            data_y[data_y == "22pestling"] = 22
            data_y[data_y == "23filtrate"] = 23
            data_y[data_y == "24catalysator"] = 24
            data_y[data_y == "25pouring"] = 25
            data_y[data_y == "26detect"] = 26
            data_y[data_y == "27end"] = 27
        elif pred_type == 'actions':  # Labels for actions are adjusted
            data_y[data_y == "cutting"] = 1
            data_y[data_y == "inverting"] = 2
            data_y[data_y == "peeling"] = 3
            data_y[data_y == "pestling"] = 4
            data_y[data_y == "pipetting"] = 5
            data_y[data_y == "pouring"] = 6
            data_y[data_y == "pour catalysator"] = 6
            data_y[data_y == "stirring"] = 7
            data_y[data_y == "transfer"] = 8
    elif dataset == 'sbhar':
        data_y[data_y == 'walking'] = 1
        data_y[data_y == 'walking_upstairs'] = 2
        data_y[data_y == 'walking_downstairs'] = 3
        data_y[data_y == 'sitting'] = 4
        data_y[data_y == 'standing'] = 5
        data_y[data_y == 'lying'] = 6
        data_y[data_y == 'stand-to-sit'] = 7
        data_y[data_y == 'sit-to-stand'] = 8
        data_y[data_y == 'sit-to-lie'] = 9
        data_y[data_y == 'lie-to-sit'] = 10
        data_y[data_y == 'stand-to-lie'] = 11
        data_y[data_y == 'lie-to-stand'] = 12
    elif dataset == 'rwhar' or dataset == 'rwhar_3sbjs':
        data_y[data_y == 'climbing_down'] = 0
        data_y[data_y == 'climbing_up'] = 1
        data_y[data_y == 'jumping'] = 2
        data_y[data_y == 'lying'] = 3
        data_y[data_y == 'running'] = 4
        data_y[data_y == 'sitting'] = 5
        data_y[data_y == 'standing'] = 6
        data_y[data_y == 'walking'] = 7
    elif dataset == 'hhar':
        data_y[data_y == 'bike'] = 1
        data_y[data_y == 'sit'] = 2
        data_y[data_y == 'stand'] = 3
        data_y[data_y == 'walk'] = 4
        data_y[data_y == 'stairsup'] = 5
        data_y[data_y == 'stairsdown'] = 6
    elif dataset == 'opportunity' or dataset == 'opportunity_ordonez' or dataset == 'opportunity_adl' or dataset == 'opportunity_drill':
        if pred_type == 'locomotion':
            data_y[data_y == "stand"] = 1
            data_y[data_y == "walk"] = 2
            data_y[data_y == "sit"] = 3
            data_y[data_y == "lie"] = 4
        elif pred_type == 'gestures':
            data_y[data_y == 'open_door_1'] = 1
            data_y[data_y == 'open_door_2'] = 2
            data_y[data_y == 'close_door_1'] = 3
            data_y[data_y == 'close_door_2'] = 4
            data_y[data_y == 'open_fridge'] = 5
            data_y[data_y == 'close_fridge'] = 6
            data_y[data_y == 'open_dishwasher'] = 7
            data_y[data_y == 'close_dishwasher'] = 8
            data_y[data_y == 'open_drawer_1'] = 9
            data_y[data_y == 'close_drawer_1'] = 10
            data_y[data_y == 'open_drawer_2'] = 11
            data_y[data_y == 'close_drawer_2'] = 12
            data_y[data_y == 'open_drawer_3'] = 13
            data_y[data_y == 'close_drawer_3'] = 14
            data_y[data_y == 'clean_table'] = 15
            data_y[data_y == 'drink_from_cup'] = 16
            data_y[data_y == 'toggle_switch'] = 17
    return data_y
