import argparse
from configparser import ConfigParser


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--c", "--configParserFile", dest="config_file", default="config.ini", type=str,
                        help="path to config file")

    args = parser.parse_args()
    config_file = args.config_file
    config = ConfigParser()
    config.read(config_file)

    parser.add_argument("--icarus_http", dest="icarusHTTP", default=config["icarusHTTP"]["http"], type=str,
                        help="transfer protocol for ICARUS camera")
    parser.add_argument("--model_weights", dest="modelPath", default=config["modelPath"]["path"], type=str,
                        help="path to model weights")
    parser.add_argument("--save_frames_path", dest="framesPath", default=config["framesPath"]["path"], type=str,
                        help="save path for grabbed frames")
    parser.add_argument("--save_grabbed_frames_path", dest="grabberPath", default=config["grabberPath"]["path"], type=str,
                        help="save path for grabbed frames")
    parser.add_argument("--max_frame_folders", dest="maxFolderNum", default=config["maxFolderNum"]["maxNumber"], type=int,
                        help="maximum folder number for frames")
    parser.add_argument("--max_frames", dest="maxFramesNum", default=config["maxFramesNum"]["maxNumber"], type=int,
                        help="maximum number of frames")
    parser.add_argument("--class_names", dest="classNames", default=config["classNames"]["names"], type=str,
                        help="names of classes")
    parser.add_argument("--coco_yaml_file", dest="cocoPath", default=config["cocoPath"]["path"], type=str,
                        help="path to COCO yaml file")

    return parser.parse_known_args()[0] if known else parser.parse_args()
