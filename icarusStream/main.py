import os
import cv2
import time
import yaml
import keyboard
from datetime import date
from args import parse_opt
from hubconf import custom


class openStream:

    def __init__(self):
        self.timer = time.time()
        self.local_time = time.localtime()
        self.cap = 0
        self.ret = 0
        self.frame = 0
        self.frameRate = 10
        self.prev = 0

    def connectCam(self, url):
        self.cap = cv2.VideoCapture(0)

    def viewStream(self, showStream=True):
        self.ret, self.frame = self.cap.read()

        self.prev = time.time()

        if showStream:
            cv2.imshow("ICARUS", self.frame)

        return self.ret, self.frame, self.cap


class frameSave:

    def __init__(self):
        self.counter = 0
        self.grabCounter = 0
        self.folderCounter = 1

    def frameGrabber(self, savePath, frame):
        try:
            if keyboard.is_pressed("s"):
                if len(os.listdir(savePath)) == 0:
                    path = os.path.join(savePath, f"frame_{str(self.grabCounter).zfill(6)}.png")
                    cv2.imwrite(path, frame)
                    self.grabCounter += 1
                else:
                    latest_folder = frameGrab.get_newest_folder(savePath)
                    pathSplit = str.split(latest_folder, "\\")
                    update_num = str.split(pathSplit[-1], "_")
                    update_num = str.split(update_num[-1], ".")
                    update_num = int(update_num[0]) + 1
                    path = os.path.join(savePath, f"frame_{str(update_num).zfill(6)}.png")
                    cv2.imwrite(path, frame)

        except:
            pass

    def saveFrames(self, savePath, frame, t, maxFolders, maxFrames, save=False):
        if save:
            folder_dir = os.listdir(savePath)
            if len(folder_dir) < maxFolders:
                subfolder = os.path.join(savePath,
                                         todayDate.strftime("%Y_%d_%m") + "_" + time.strftime("%H_%M_%S", t))
                if not os.path.exists(subfolder):
                    os.mkdir(subfolder)
                subfolder_dir = os.listdir(subfolder)
                if len(subfolder_dir) < maxFrames:
                    imagePath = os.path.join(subfolder, f"frame_{str(self.counter).zfill(6)}.png")
                    cv2.imwrite(imagePath, frame)
                    self.counter += 1
                else:
                    t = time.localtime()
                    self.counter = 0

        return t * save, self.counter * save

    def get_all_folders(self, path):
        return [x for x in os.listdir(path) if os.path.isfile(x) is False]

    def get_newest_folder(self, path):
        newest = None
        date = None

        for f in frameGrab.get_all_folders(path):
            file = os.path.join(path, f)
            if date is None or date < os.path.getmtime(file):
                newest = file
                date = os.path.getmtime(file)

        return os.path.join(path, newest)


def convert2Bbox(box, img_shape):
    cx = int((box[0] - box[2] / 2) * img_shape[1])
    cy = int((box[1] - box[3] / 2) * img_shape[0])
    w = int(box[2] * img_shape[1])
    h = int(box[3] * img_shape[0])

    return cx, cy, w, h


def myModel(modelPath):
    model_name = str.split(modelPath, "\\")
    model_name = model_name[-1]

    model = custom(modelPath)

    return model, model_name


def coco(yamlFile):
    COCOclasses = []
    with open(yamlFile, 'r', encoding="utf-8") as file:
        prime_service = yaml.safe_load(file)
        for k, v in prime_service.items():
            if k == "names":
                for idx, name in v.items():
                    COCOclasses.append(name)
    return COCOclasses


def plotBoxes(preds):
    for box in preds[0]:
        x, y, w, h = convert2Bbox(box[:4], frame.shape[:2])
        conf = round(box[-2].item(), 2)
        label = int(box[-1].item())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        (width, height), _ = cv2.getTextSize(class_labels[label] + ": " + str(conf), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if y <= 10:
            cv2.rectangle(frame, (x-1, y+h-2), (x+width, y+height+h), (0, 0, 255), -1)
            cv2.putText(frame, class_labels[label] + ": " + str(conf), (x, y+h+height-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (x - 1, y-height-2), (x + width, y), (0, 0, 255), -1)
            cv2.putText(frame, class_labels[label] + ": " + str(conf), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)


if __name__ == "__main__":
    opt = parse_opt()
    icarusCam = openStream()
    icarusCam.connectCam(url=opt.icarusHTTP)
    yolo, modelName = myModel(opt.modelPath)
    frameGrab = frameSave()
    todayDate = date.today()
    localTime = time.localtime()
    frame_rate = 15
    prev = 0

    if modelName == "yolov5l.pt":
        class_labels = coco(opt.cocoPath)
        len_classes = len(class_labels)
    else:
        len_classes = len(opt.classNames.split(", "))
        class_labels = opt.classNames.split(", ")

    while True:
        ret, frame, cap = icarusCam.viewStream(showStream=False)

        if not ret:
            print("Cannot receive frame")
            break

        localTime, image_counter = frameGrab.saveFrames(opt.framesPath,
                                                        frame=frame,
                                                        t=localTime,
                                                        maxFolders=opt.maxFolderNum,
                                                        maxFrames=opt.maxFramesNum,
                                                        save=False)

        start = time.time()
        prediction = yolo(frame).xywhn
        end = time.time() - start
        print(end)
        plotBoxes(preds=prediction)
        frameGrab.frameGrabber(opt.grabberPath, frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
