import os
import numpy as np
import random
from wsi_op import read_wsi, find_roi_bbox
import pandas as pd
import cv2
from collections import defaultdict
import torch
import h5py
import os
from sklearn.model_selection import train_test_split

def load_files(dataset_path):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """
    dataset = {}
    dataset['train'] = [os.path.join(root,file) for root,dirs, files in os.walk(os.path.join(dataset_path, "training")) for file in files if file.endswith(".h5")]
    dataset['test'] = [os.path.join(root,file) for root,dirs, files in os.walk(os.path.join(dataset_path, "testing")) for file in files if file.endswith(".h5")]

    return dataset


def Get_train_valid_Path(Train_set, train_percentage=0.9):
    # random.seed(12321)
    # indexes = np.arange(len(Train_set))
    # random.shuffle(indexes)
    #
    # num_train = int(train_percentage * len(Train_set))
    # train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])
    #
    # Model_Train = [Train_set[i] for i in train_index]
    # Model_Val = [Train_set[j] for j in test_index]

    #Model_Train,Model_Val = train_test_split(Train_set, test_size = 0.1, random_state = 12321,stratify=True)
    train_labels = [int(1) if "tumor" in os.path.splitext(os.path.basename(path))[0] else int(0) for path in Train_set]
    Model_Train, Model_Val, y_train, y_test = train_test_split(Train_set, train_labels, test_size=1-train_percentage,
                                                        random_state=12321, stratify=train_labels)

    return Model_Train, Model_Val


def detect_tissue(image_path):

            img_name, wsi_image, rgb_image, level_used = read_wsi(image_path)
            assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

            rgba_image, bounding_boxes, rgb_contour, image_open = find_roi_bbox(rgb_image)

            return wsi_image,bounding_boxes,image_open,level_used


def load_bags(wsi_path, train):


        class_name = os.path.basename(wsi_path).split(".")[0]
        coords=[]
        if train:
            bag_label=(int(1) if "tumor" in class_name else int(0))
        else:
            references=pd.read_csv(csv_file,header=None)
            label=references[1].loc[references[0]==class_name].values.tolist()[0]
            bag_label = (int(1) if "Tumor" in label else int(0))

        with h5py.File(wsi_path, "r") as hdf5_file:

                h5_coords = hdf5_file['coords'][:]
                coords.append(h5_coords)

        return coords, int(bag_label)

def create_dict(filenames, level, train, patch_size, stride, dict_name,csv_file=None):

        d = defaultdict()
        file_list=[]
        coord_list=[]
        label_list=[]
        for file in filenames:

            coords, label=load_bags(file,train=train, csv_file=csv_file)
            file_list.append(file)
            coord_list.append(coords)
            label_list.append(label)

            d["slides"]= file_list
            d["grid"]  = coord_list
            d["targets"]= label_list
            d["mult"]= 1
            d["level"]= level
        torch.save(d, '{}.pt'.format(dict_name))
        return d

if __name__ == '__main__':

    csv_file = "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/testing/reference.csv"
    dataset_path = "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16/Nature-2019-patches"

    dataset = load_files(dataset_path=dataset_path)

    train_bags = dataset['train']
    train_bags, valid_bags=Get_train_valid_set(train_bags)
    test_bags = dataset['test']

    create_dict(train_bags, csv_file=None,dict_name="train_dict")
    create_dict(valid_bags, csv_file=None, dict_name="valid_dict")
    create_dict(test_bags,  csv_file=csv_file, dict_name="test_dict")



