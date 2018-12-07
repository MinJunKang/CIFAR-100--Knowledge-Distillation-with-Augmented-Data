# Python Header - Import dir



try:
    import os
    import sys
    import argparse
    import numpy as np
    import random
    import shutil
    import pickle
    import matplotlib.pyplot as plt
    import warnings
    import math
    import time

    # Randomness setting
    seed = int(time.time())
    np.random.seed(seed)
    random.seed(seed)

except Exception as e:
    print("Error : ",e)
    exit(1)

# import tensorflow and keras
try:
    import tensorflow as tf
    import keras
    import keras.applications as ap
    import keras.backend as K
    import keras.activations as act
    import keras.utils as ut
    import keras.callbacks as call
    import keras.optimizers as op
    import keras.layers as l
    import keras.models as m
    import keras.regularizers as rz
    import keras.initializers as lz
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import np_utils

    tf.set_random_seed(seed)

except Exception as e:
    print("Error : ",e)
    exit(1)

# for no warning sign
warnings.filterwarnings('ignore')

# Temperature storage
temp_dir = "./temp/"
temp_file = "./temp/temperature.pickle"

# GPU SETUP
def set_up(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (args.gpu)

def Set_Model_Info():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # for hardware acceleration
    parser.add_argument('--gpu', type=int, default=None, choices=[None, 0, 1])

    # Training Parameters
    parser.add_argument('--model_usage', type=str, default="teacher", choices=["teacher", "student"])

    # Language
    parser.add_argument('--model_lang',type=str,default="keras",choices=["keras","tensorflow"])

    return parser


def Set_Parameter(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Temperature
    parser.add_argument('--temperature', type=float, default=1.0)

    # Set epoch and stop point criterion
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--max_overfit',type=int,default=50)

    # Model Type
    parser.add_argument('--model_type',type=str,default="classifier",choices=["classifier", "regression"])

    ## Learning Parameters ##

    # Set learning rate
    parser.add_argument('--min_learning_rate',type=float,default=0.0001)
    parser.add_argument('--learning_rate_increment',type=float,default=10)
    parser.add_argument('--max_learning_rate',type=float,default=0.0001)

    # Set batch size
    parser.add_argument('--min_batch', type=int, default=128)
    parser.add_argument('--batch_increment', type=int, default=2)
    parser.add_argument('--max_batch', type=int, default=128)

    if(args.model_usage == "teacher"):
        # Learning parameter
        # parser.add_argument('--dropout',type=float,default=0.3)

        # Model information
        parser.add_argument('--input_shape',type=list,default=[224,224,3])
        parser.add_argument('--output_shape',type=list,default=[100])

        # Save path
        parser.add_argument('--dir',type=str,default="./Teacher_model/")
        parser.add_argument('--checkpoint',type=str,default="./Teacher_model/checkpoint/") # checkpoint weight saved
        parser.add_argument('--model',type=str,default="./Teacher_model/model/") # model saved
        parser.add_argument('--final',type=str,default="./Teacher_model/final/") # final weight saved
        parser.add_argument('--log',type=str,default="./Teacher_model/log/") # log dir for softy data
        parser.add_argument('--softy_file',type=str,default="./Teacher_model/log/soft_y.pickle") # log dir for softy data

    else:

        ## Model Parameters ##
    
        # DNN Node Setting
        parser.add_argument('--min_node',type=int,default=32)
        parser.add_argument('--node_increment',type=int,default=2)
        parser.add_argument('--max_node',type=int,default=64)

        # Layer Setting
        parser.add_argument('--min_layer',type=int,default = 2)
        parser.add_argument('--layer_increment',type=int,default = 1)
        parser.add_argument('--max_layer',type=int,default = 4)

        # Save path
        parser.add_argument('--dir',type=str,default="./Student_model/")
        parser.add_argument('--checkpoint',type=str,default="./Student_model/checkpoint/") # checkpoint weight saved
        parser.add_argument('--model',type=str,default="./Student_model/model/") # model saved
        parser.add_argument('--final',type=str,default="./Student_model/final/") # final weight saved
        parser.add_argument('--log',type=str,default="./Student_model/log/") # log dir for recording models
        parser.add_argument('--tmp',type=str,default="./Student_model/tmp/") # tmp dir for simulating single model
        parser.add_argument('--softy_file',type=str,default="./Teacher_model/log/soft_y.pickle") # log dir for softy data
        parser.add_argument('--teacher_acc_file',type=str,default="./Teacher_model/log/Accuracy.pickle") # log dir for teacher accuracy

    return parser




# Global Function

def check_and_makedir(path,choice=False):
    if(choice==True):
        shutil.rmtree(path, ignore_errors=True)
    if not os.path.isdir(path):
        os.makedirs(path)

def Save_Text(contents,dst_file,mode = 'wb'):
    with open(dst_file,mode) as mysavedata:
        pickle.dump(contents,mysavedata)
    return True

def Load_Text(dst_file):
    contents = []
    with open(dst_file,'rb') as myloaddata:
        while(1):
            try:
                contents = contents + pickle.load(myloaddata)
            except EOFError:
                break
    return contents

# Shuffle the data
def Shuffle_Data(pair_1,pair_2):
    # Shuffle the data
    overall_data = list(zip(pair_1,pair_2))
    random.shuffle(overall_data)
    result_1 = []
    result_2 = []
    for i in range(len(overall_data)):
        result_1.append(overall_data[i][0])
        result_2.append(overall_data[i][1])

    return result_1,result_2

def dlProgress(count, blockSize, totalSize):
    percent = int(count*blockSize*100/totalSize)
    sys.stdout.write("\r" + "Downloading process...%d%%" % percent)
    sys.stdout.flush()

def dlProgress_2(count,total,start_time):
    now = time.time()
    time_left = (now - start_time) / float(count / (total-count))
    final = time.gmtime(time_left + now + 9*3600)
    sys.stdout.write("\r" + "Process...%d%% ( %d / %d ) [ Exit Time : %d / %d / %d   %d : %d : %d ]" % (int(count * 100 / total),count,total,final.tm_year,final.tm_mon,final.tm_mday,final.tm_hour, final.tm_min,final.tm_sec))
    sys.stdout.flush()




