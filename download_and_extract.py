import os
import mlflow
import argparse

def download_and_extract(kaggle_url, out_path):
    os.system('cp kaggle.json /home/nolan/.kaggle/kaggle.json')
    os.system('chmod 600 /home/nolan/.kaggle/kaggle.json')
    os.system(kaggle_url)
    os.makedirs(out_path, exist_ok = True)
    os.system('unzip aptos2019-blindness-detection.zip -d '+ out_path)
    os.system('rm aptos2019-blindness-detection.zip')

if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle_url", default="kaggle competitions download -c aptos2019-blindness-detection", type = str, help= "The kaggle competiions dataset download syntax (kaggle cli format)")
    parser.add_argument("--out_path", default="dataset", type = str, help= "The output path in which the downloaded datasets contants will be stored")
    args = parser.parse_args()
    download_and_extract(args.kaggle_url, args.out_path)