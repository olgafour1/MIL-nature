import os
import numpy as np
import random
from wsi_op import read_wsi, find_roi_bbox
import pandas as pd
import cv2
from collections import defaultdict
import torch

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
    dataset['train'] = [os.path.join(root,file) for root,dirs, files in os.walk(os.path.join(dataset_path, "training")) for file in files if file.endswith(".tif")]
    dataset['test'] = [os.path.join(root,file) for root,dirs, files in os.walk(os.path.join(dataset_path, "testing")) for file in files if file.endswith(".tif")]

    return dataset

def Get_train_valid_set(Train_set, train_percentage=0.9):
    indexes = np.arange(len(Train_set))
    random.shuffle(indexes)

    num_train = int(train_percentage * len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val


def detect_tissue(image_path):

            img_name, wsi_image, rgb_image, level_used = read_wsi(image_path)
            assert wsi_image is not None, 'Failed to read Whole Slide Image %s.' % image_path

            rgba_image, bounding_boxes, rgb_contour, image_open = find_roi_bbox(rgb_image)

            return wsi_image,bounding_boxes,image_open,level_used


def load_bags(wsi_path, level,train, patch_size, stride,csv_file=None):

        wsi_image,bounding_boxes,image_open,level_used=detect_tissue(wsi_path)

        mag_factor_used = pow(2, level_used - level)
        mag_factor = pow(2, level)

        class_name = os.path.basename(wsi_path).split(".")[0]
        coords=[]
        if train:
            bag_label=(int(1) if "tumor" in class_name else int(0))
        else:
            references=pd.read_csv(csv_file,header=None)
            label=references[1].loc[references[0]==class_name].values.tolist()[0]
            bag_label = (int(1) if "Tumor" in label else int(0))

        for bounding_box in bounding_boxes:

                                b_x_start = int(bounding_box[0])*mag_factor_used
                                b_y_start = int(bounding_box[1])*mag_factor_used
                                b_x_end = (int(bounding_box[0]) + int(bounding_box[2]))*mag_factor_used
                                b_y_end = (int(bounding_box[1]) + int(bounding_box[3]))*mag_factor_used

                                X = np.arange(b_x_start, b_x_end, step=stride)
                                Y = np.arange(b_y_start, b_y_end, step=stride)

                                for w_pos, x_width_ in enumerate(X):
                                    for h_pos, y_height_ in enumerate(Y):

                                                x_coord = x_width_ * mag_factor
                                                y_coord = y_height_ * mag_factor

                                                patch = wsi_image.read_region((x_coord, y_coord),level,
                                                                              (patch_size, patch_size))

                                                patch_array = np.array(patch)

                                                patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)

                                                lower_red = np.array([20, 20, 20])

                                                upper_red = np.array([200, 200, 200])

                                                mask = cv2.inRange(patch_hsv, lower_red, upper_red)

                                                white_pixel_cnt = cv2.countNonZero(mask)

                                                if white_pixel_cnt >= ((patch_size * patch_size) * 0.5):

                                                    coords.append((int(((x_width_) + (patch_size / 2))), int(((y_height_) + (patch_size / 2)))))

        return coords, int(bag_label)

def create_dict(filenames, level, train, patch_size, stride, dict_name,csv_file=None):

        d = defaultdict()
        file_list=[]
        coord_list=[]
        label_list=[]
        for file in filenames:

            coords, label=load_bags(file, level=level, train=train, patch_size=stride, stride=patch_size, csv_file=csv_file)
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
    dataset_path = "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/camelyon16"

    dataset = load_files(dataset_path=dataset_path)

    train_bags = dataset['train']
    train_bags, valid_bags=Get_train_valid_set(train_bags)
    test_bags = dataset['test']

    #create_dict(train_bags, level=1, train=True, patch_size=224, stride=224, csv_file=None,dict_name="train_dict")
    #create_dict(valid_bags, level=1, train=True, patch_size=224, stride=224, csv_file=None, dict_name="valid_dict")
    create_dict(test_bags, level=1, train=False, patch_size=224, stride=224, csv_file=csv_file, dict_name="test_dict")



