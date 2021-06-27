import os
import shutil
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_validation_split(data_path):
    split = []
    scores =  pd.read_csv(data_path+'/train.csv')
    _ , val_idx = train_test_split(np.arange(len(scores.id_code)), test_size=0.2, shuffle=True, stratify= scores.diagnosis)
    if not os.path.isdir(data_path+'/val_images'):
        os.mkdir(data_path+'/val_images')
    for i, img in enumerate(scores.id_code):
        if i in val_idx:
            shutil.move(data_path+'/train_images/'+img+".png", data_path+'/val_images/'+img+".png")
            split.append('val')
        else:
            split.append('train')
    scores['split'] = split
    scores.to_csv(data_path+'/train_val.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default = "dataset", type = str, help= "the path where the downloaded dataset is stored")
    args = parser.parse_args()
    train_validation_split(args.data_path)
