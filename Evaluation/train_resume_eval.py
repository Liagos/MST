import os
import re
import sys
import wandb
import pickle
import shutil
import subprocess
import numpy as np
import configparser
from args import parse_opt
import inference_tflite_V2 as infer
from inference_args import parse_opt_inference


def updateConfigFile(modelPath, imagesPath, labelsPath, outputDir):
    configFile = "config_infer.ini"
    config_file = configparser.ConfigParser()
    config_file.read(configFile)

    if not config_file.has_section("modelname"):
        config_file.add_section("modelname")
    config_file.set("modelname", 'path', modelPath)
    if not config_file.has_section("testimages"):
        config_file.add_section("testimages")
    config_file.set("testimages", 'path', imagesPath)
    if not config_file.has_section("testlabels"):
        config_file.add_section("testlabels")
    config_file.set("testlabels", 'path', labelsPath)
    if not config_file.has_section("outputDir"):
        config_file.add_section("outputDir")
    config_file.set("outputDir", 'path', outputDir)

    with open('config_infer.ini', 'w') as configfile:
        config_file.write(configfile)


class resume_train:
    def get_all_folders(self, path):
        return [x for x in os.listdir(path) if os.path.isfile(x) is False]

    def get_newest_folder(self, path):
        newest = None
        date = None

        for f in resume_training.get_all_folders(path):
            file = os.path.join(path, f)
            if date is None or date < os.path.getmtime(file):
                newest = file
                date = os.path.getmtime(file)

        return os.path.join(path, newest)

    def get_weights(self):
        latest_folder = resume_training.get_newest_folder(opt.runsPath)
        weights_path = os.listdir(os.path.join(latest_folder, "weights"))
        weights_path = sorted(
            [os.path.join(latest_folder, "weights", weight) for weight in weights_path if weight.startswith("epoch")])

        return weights_path

    def train(self, n_e, b_s, train_model=False):
        if train_model:
            wandb.login(key="d865eb038d328ee5a99df7151fc71324b3837500")

            yaml_train_file = opt.yamlTrain
            subprocess.run([sys.executable, opt.yoloPath + "/train.py",
                            "--batch-size", str(b_s),
                            "--epochs", str(n_e),
                            "--data", str(yaml_train_file),
                            "--weights", "yolov5l.pt",
                            "--freeze", "10",
                            "--name", "coco_theo_icarus_backbone_freeze_weighted",
                            "--save-period", "1",
                            "--image-weights"])


class validation:

    def rename_weights(self):
        latest_folder = resume_training.get_newest_folder(opt.runsPath)
        weights_path = os.path.join(opt.runsPath, latest_folder, "weights")
        weights_dir = [weight for weight in sorted(os.listdir(weights_path)) if weight.startswith("epoch")]
        for weight in weights_dir:
            split_weight = str.split(weight, ".")
            temp = re.compile('([a-zA-Z]+)([0-9]+)')
            try:
                res = temp.match(split_weight[0]).groups()
            except:
                continue
            new_name = f"{res[0]}_{res[1].zfill(4)}.pt"
            src = os.path.join(weights_path, weight)
            dst = os.path.join(weights_path, new_name)
            os.rename(src, dst)

    def validate_model(self, validate_model=False):
        if validate_model:
            model_weights = resume_training.get_weights()
            yaml_files = str.split(opt.yamlFiles, ", ")
            for model in model_weights:
                spit_path = str.split(model, "/")
                model_name = spit_path[-1]
                split_weight = str.split(model_name, "_")
                current_epoch = int(str.split(split_weight[-1], ".")[0])
                for yaml in yaml_files:
                    name = str.split(yaml, "_")
                    folder_path = os.path.join(opt.yoloPath, "runs", "val", f"{str(name[0])}_epochs_{current_epoch}")
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                    subprocess.run([sys.executable, opt.yoloPath + "/val.py",
                                    "--weights", str(model),
                                    "--data", str(yaml),
                                    "--name", f"{str(name[0])}_epochs_{current_epoch}",
                                    "--save-txt",
                                    "--save-conf"])
                current_epoch += 1


class evaluate_inference:
    def __init__(self):
        datasets = ["COCO", "FES", "ICARUS"]
        self.precision_dict = {dataset: [] for dataset in datasets}
        self.recall_dict = {dataset: [] for dataset in datasets}
        self.scores_dict = {dataset: [] for dataset in datasets}

    def eval_script(self):
        opt_infer = parse_opt_inference()
        model_weights = resume_training.get_weights()
        test_data_paths = str.split(opt.testData, ", ")
        suffix = ["COCO", "FES", "ICARUS"]
        pretrainedModel = os.path.join(opt.yoloPath, "yolov5l.pt")
        for path in test_data_paths:
            updateConfigFile(pretrainedModel, path + "/images", path + "/labels", opt_infer.outputDir)
            res = infer.main()
            self.precision_dict[suffix[test_data_paths.index(path)]].append(res[0])
            self.recall_dict[suffix[test_data_paths.index(path)]].append(res[1])
            self.scores_dict[suffix[test_data_paths.index(path)]].append(res[2])
            # #     results = subprocess.check_output([sys.executable, opt.yoloPath + "/inference_tflite_V2.py",
            # #                               "-m", "yolov5l.pt",
            # #                               "-t", path + "/images",
            # #                               "-l", path + "/labels"])
            latest_eval = resume_training.get_newest_folder(opt_infer.outputDir)
            dst_folder = f"{latest_eval}_{suffix[test_data_paths.index(path)]}_epochs_{-1}"
            os.rename(latest_eval, dst_folder)

        for model in model_weights:
            spit_path = str.split(model, "/")
            model_name = spit_path[-1]
            split_weight = str.split(model_name, "_")
            current_epoch = int(str.split(split_weight[-1], ".")[0])
            for path in test_data_paths:
                updateConfigFile(model, path + "/images", path + "/labels", opt_infer.outputDir)
                res = infer.main()
                self.precision_dict[suffix[test_data_paths.index(path)]].append(res[0])
                self.recall_dict[suffix[test_data_paths.index(path)]].append(res[1])
                self.scores_dict[suffix[test_data_paths.index(path)]].append(res[2])
                # subprocess.run([sys.executable, opt.yoloPath + "/inference_tflite_V2.py",
                #                 "-m", str(model),
                #                 "-t", path + "/images",
                #                 "-l", path + "/labels"])
                latest_eval = resume_training.get_newest_folder(opt_infer.outputDir)
                dst_folder = f"{latest_eval}_{suffix[test_data_paths.index(path)]}_epochs_{current_epoch}"
                os.rename(latest_eval, dst_folder)
            current_epoch += 1

        return self.precision_dict, self.recall_dict, self.scores_dict

    def save_precison_recall_score(self, folder, precision_dict, recall_dict, score_dict):
        pr_path = os.path.join(folder, "precision_dict.pkl")
        re_path = os.path.join(folder, "recall_dict.pkl")
        sc_path = os.path.join(folder, "scores_dict.pkl")

        pr_file = open(pr_path, "wb")
        pickle.dump(precision_dict, pr_file)
        pr_file.close()
        re_file = open(re_path, "wb")
        pickle.dump(recall_dict, re_file)
        re_file.close()
        scores_file = open(sc_path, "wb")
        pickle.dump(score_dict, scores_file)
        scores_file.close()


if __name__ == "__main__":
    opt = parse_opt()
    resume_training = resume_train()
    validate = validation()
    eval_detection = evaluate_inference()
    resume_training.train(n_e=opt.numEpochs,
                          b_s=opt.batchSize,
                          train_model=False)
    validate.rename_weights()
    validate.validate_model(validate_model=False)
    pr_dict, re_dict, scores_dict = eval_detection.eval_script()
    eval_detection.save_precison_recall_score(opt.dictsPath, pr_dict, re_dict, scores_dict)
