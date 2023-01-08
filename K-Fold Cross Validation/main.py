import os
import sys
import yaml
import wandb
import shutil
import seaborn
import subprocess
import matplotlib
import torchvision
import pandas as pd
from tqdm import tqdm
from args import parse_opt
from shutil import copyfile
from sklearn.model_selection import StratifiedKFold

wandb.login(key="d865eb038d328ee5a99df7151fc71324b3837500")


def train_fold_csv(csv_file_path, num_folds, use_fold=False):
    if "train_fold.csv" in os.listdir(csv_file_path) and use_fold:
        df = pd.read_csv(os.path.join(csv_file_path, "train_fold.csv"), header=None)
    else:
        df = pd.read_csv(os.path.join(csv_file_path, "dataset.csv"), header=None)
        Fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        for n, (train_index, val_index) in enumerate(Fold.split(df.iloc[:, 0], df.iloc[:, 2])):
            df.loc[val_index, "fold"] = int(n)
        df["fold"] = df["fold"].astype(int)
        df.to_csv(os.path.join(csv_file_path, "train_fold.csv"), index=False)

    return df


def makeYamls(csv_file_path, num_folds, yoloPath, classes):
    names_dict = {}
    for fold in range(num_folds):
        for idx, c in enumerate(classes):
            names_dict[idx] = c
        data_yaml = dict(
            path=csv_file_path,
            train=f'dataset_folds_{fold}/images/train',
            val=f'dataset_folds_{fold}/images/val',
            names=names_dict)
        yolo_path = os.path.join(yoloPath, f'data/data_fold_{fold}.yaml')
        with open(yolo_path, 'w', newline="") as outfile:
            yaml.dump(data_yaml, outfile, default_flow_style=False, sort_keys=False)


def makeFolds(num_folds, folds_csv, csv_file_path, n_e, b_s, img_s, yolo, use_fold=False):
    if use_fold:
        pass
    else:
        for fold in range(num_folds):
            train_df = folds_csv.loc[folds_csv.fold != fold].reset_index(drop=True)
            val_df = folds_csv.loc[folds_csv.fold == fold].reset_index(drop=True)

            try:
                img_fold_path = os.path.join(csv_file_path, f"dataset_folds_{fold}/images")
                lbl_fold_path = os.path.join(csv_file_path, f"dataset_folds_{fold}/labels")
                shutil.rmtree(img_fold_path)
                shutil.rmtree(lbl_fold_path)
            except:
                print('No dirs')

            os.makedirs(os.path.join(csv_file_path, f'dataset_folds_{fold}/images/train'), exist_ok=True)
            os.makedirs(os.path.join(csv_file_path, f'dataset_folds_{fold}/images/val'), exist_ok=True)
            os.makedirs(os.path.join(csv_file_path, f'dataset_folds_{fold}/labels/train'), exist_ok=True)
            os.makedirs(os.path.join(csv_file_path, f'dataset_folds_{fold}/labels/val'), exist_ok=True)

            for i in tqdm(range(len(train_df))):
                img_file_source = os.path.join(train_df.iloc[i, 0], train_df.iloc[i, 1])
                img_file_dest = os.path.join(csv_file_path,
                                             f'dataset_folds_{fold}/images/train/{train_df.iloc[i, 1]}')
                lbl_file_source = os.path.join(train_df.iloc[i, 2], train_df.iloc[i, 3])
                lbl_file_dest = os.path.join(csv_file_path,
                                             f'dataset_folds_{fold}/labels/train/{train_df.iloc[i, 3]}')
                copyfile(img_file_source, img_file_dest)
                copyfile(lbl_file_source, lbl_file_dest)

            for i in tqdm(range(len(val_df))):
                img_file_source = os.path.join(val_df.iloc[i, 0], val_df.iloc[i, 1])
                img_file_dest = os.path.join(csv_file_path,
                                             f'dataset_folds_{fold}/images/val/{val_df.iloc[i, 1]}')
                lbl_file_source = os.path.join(val_df.iloc[i, 2], val_df.iloc[i, 3])
                lbl_file_dest = os.path.join(csv_file_path, f'dataset_folds_{fold}/labels/val/{val_df.iloc[i, 3]}')
                copyfile(img_file_source, img_file_dest)
                copyfile(lbl_file_source, lbl_file_dest)

            print(f"****************************FOLD: {fold}****************************")
            subprocess.run([sys.executable, yolo+"/train.py",
                            "--batch-size", str(b_s),
                            "--epochs", str(n_e),
                            "--data", f"data_fold_{fold}.yaml",
                            "--weights", "yolov5l.pt",
                            "--project", "YOLOv5_K-Fold",
                            "--freeze", "9",
                            "--name", f"yolov5l-fold-{fold}"])

            print(f"****************************END FOLD: {fold}****************************")

            img_lbl_fold_path = os.path.join(csv_file_path, f"dataset_folds_{fold}")

            shutil.rmtree(img_lbl_fold_path)


if __name__ == "__main__":
    opt = parse_opt()
    dataFrame = train_fold_csv(csv_file_path=opt.csvPath, num_folds=opt.numFolds, use_fold=False)

    # makeYamls(csv_file_path=opt.csvPath,
    # num_folds=opt.numFolds,
    # yoloPath=opt.yoloPath,
    # classes=", ".join(opt.classNames))

    makeFolds(num_folds=opt.numFolds,
              csv_file_path=opt.csvPath,
              folds_csv=dataFrame,
              b_s=opt.batchSize,
              n_e=opt.numEpochs,
              img_s=opt.imageSize,
              yolo=opt.yoloPath,
              use_fold=False)
