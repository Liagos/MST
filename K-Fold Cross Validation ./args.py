import argparse
from configparser import ConfigParser


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--configParserFile", dest='config_file', default='config.ini', type=str,
                        help="path to config file for the annotation parser")

    args = parser.parse_args()

    config_file = args.config_file
    config = ConfigParser()
    config.read(config_file)

    parser.add_argument("--csvPath", dest="csvPath", default=config["csvPath"]["path"], type=str,
                        help="path to dataset csv file")
    parser.add_argument("--img_size", dest="imageSize", default=config["imageSize"]["size"], type=int,
                        help="size of input image")
    parser.add_argument("--batch_size", dest="batchSize", default=config["batchSize"]["size"], type=int,
                        help="batch size")
    parser.add_argument("--num_epochs", dest="numEpochs", default=config["numEpochs"]["epochs"], type=int,
                        help="number of epochs")
    parser.add_argument("--num_folds", dest="numFolds", default=config["numFolds"]["folds"], type=int,
                        help="number of folds")
    parser.add_argument("--yolo_path", dest="yoloPath", default=config["yoloPath"]["path"], type=str,
                        help="path to yolov5/train.py")
    parser.add_argument("--class_names", dest="classNames", default=config["classNames"]["class"], type=str,
                        help="names of classes")
    parser.add_argument("--weights", type=str, help="initial weights path")

    return parser.parse_known_args()[0] if known else parser.parse_args()
