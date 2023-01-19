# K-Fold Cross Validation

This repository contains the code for the project "**K-fold Cross Validation**". The goal of this project is to perform a 5-fold cross validation using a YOLO V5 object detection model. 

The repository is organised as follows:
- The `main.py` file creates the train and test sets and runs perfomr the validation.
- The `args.py` file parses the arguments needed for the execution.
- The `confing.ini` file contains the csv path, class names, image size, batch size, number pf epochs, number of folds, and finaly the path to the YOLO V5 repository.

To run the code, you will need to install the libraries in the `requirements.txt` file. We recommend using [PyCharm](https://www.jetbrains.com/pycharm/promo/?source=google&medium=cpc&campaign=14123077402&term=pycharm&gclid=Cj0KCQjw6_CYBhDjARIsABnuSzqkMV4IXzjuVu-enSX0e70lwTUQBmgEFAoSE3uktD045-LG9A0s0acaAqEDEALw_wcB).

# Icarus Stream

This repository contains the code for the project "**icarusStream**". The goal of this project is to establish connection wiht a top-view omniderectional camera over HTTP. Then, the user is able to save single or series or frames. Finally, the trained model is able to make predictions on the incoming frames and plot the predicitons.

The repository is organised as follows:
- The `main.py` file establishes the connection with the camera, imports the model, makes and plots predictions.
- The `args.py` file parses the arguments needed for the execution.
- The `confing.ini` file contains the HTTP path, model path, save frames folder path, save single frames folder path, maximum number of save folders, maximum number of frames in each folder, class names, path to coco.yaml containing all 80 classes in case a pre-trained YOLO V5 is used.

To run the code, you will need to install the libraries in the `requirements.txt` file. We recommend using [PyCharm](https://www.jetbrains.com/pycharm/promo/?source=google&medium=cpc&campaign=14123077402&term=pycharm&gclid=Cj0KCQjw6_CYBhDjARIsABnuSzqkMV4IXzjuVu-enSX0e70lwTUQBmgEFAoSE3uktD045-LG9A0s0acaAqEDEALw_wcB).

All the files must be saved inside the YOLO V5 repository on the user's computer before executing the `main.py` script.

# Evaluate

This repository contains the code for the project "**Evaluate*". The goal of this project is to evaluate the performance of the model. The `inference_tflite_V2.py` is used to calculate the precision and recall of a model by comparing its predictions against corresponding ground truths.

The repository is organised as follows:
- The `train_resume_eval.py` uses the weights of a given model to evaluate its performance. The user has the option to train their model and then evaluate on a given test set.
- The `args.py` and `inference_args.py` files parse the arguments needed for the execution.
- The `confing.ini` file contains the path to YOLO V5 repository, the path to train folder, the yaml files needed for each test set, path to test data folders, save path, number of epochs, batch size, and save path for the precision and recall dictionaries.
- The `confing_infer.ini` file contains the model path, the test images path, the test labels path, and the output directory to save the evaluation results


To run the code, you will need to install the libraries in the `requirements.txt` file. We recommend using [PyCharm](https://www.jetbrains.com/pycharm/promo/?source=google&medium=cpc&campaign=14123077402&term=pycharm&gclid=Cj0KCQjw6_CYBhDjARIsABnuSzqkMV4IXzjuVu-enSX0e70lwTUQBmgEFAoSE3uktD045-LG9A0s0acaAqEDEALw_wcB).

All the files must be saved inside the YOLO V5 repository on the user's computer before executing the `train_resume_eval.py` script.

# utils

In folder utils the scripts used to plot the precision-recall curves, mAP curves, and 3D plots can be found. 
