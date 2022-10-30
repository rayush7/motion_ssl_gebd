# by htd@robots.ox.ac.uk
from joblib import delayed, Parallel
import os 
import sys 
import glob 
import subprocess
from tqdm import tqdm 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
import pickle
import pandas as pd

def resize_video_ffmpeg(vid_list,v_path, out_path, dim=256):
    '''v_path: single video path;
       out_path: root to store output videos'''
    print('********* vpath ******************',v_path)
    
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    print('------- vname -------',v_name)

    #----------------------------------------------
    # Get video id
    v_id = v_name[4:]

    print('------ video id ---------',v_id)

    if v_id in vid_list:
        print(v_id,'  ********** exists ***********')
        
    else:
        print(v_id,'  ********** not exists ***********')
        return
    #--------------------------------------------------

    out_dir = os.path.join(out_path, v_class)
    if not os.path.exists(out_dir):
        raise ValueError("directory not exist, it shouldn't happen")

    vidcap = cv2.VideoCapture(v_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        print(v_path, 'not successfully loaded, drop ..'); return
    new_dim = resize_dim(width, height, dim)
    if new_dim[0] == dim: 
        dim_cmd = '%d:-2' % dim
    elif new_dim[1] == dim:
        dim_cmd = '-2:%d' % dim 

    cmd = ['ffmpeg', '-loglevel', 'quiet', '-y',
           '-i', '%s'%v_path,
           '-vf',
           'scale=%s'%dim_cmd,
           '%s' % os.path.join(out_dir, os.path.basename(v_path))]
    ffmpeg = subprocess.call(cmd)

def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return [int(target * w / h), int(target)]
    else:
        return [int(target), int(target * h / w)]

def main_kinetics400(vid_list,output_path='your_path/kinetics400'):
    print('save to %s ... ' % output_path)
    for splitname in ['test']: # val_split,train_split
        # Avoid running on train split
        #if splitname=='val_split':
        #    continue
        #v_root = '../kinetics400/videos' + '/' + splitname
        #v_root = '/home/ayushrai/kinetics400/GEBD_kinetics400' + '/' + splitname
        v_root = '/media/ayushrai/8TB_HDD/kinetics-dataset/k400_gebd' + '/' + splitname
        if not os.path.exists(v_root):
            print('Wrong v_root')
            import ipdb; ipdb.set_trace() # for debug

        # Comment this line when running for train and test
        # splitname_new = 'test_split'   

        out_path = os.path.join(output_path, splitname) 
        if not os.path.exists(out_path): 
            os.makedirs(out_path)
        v_act_root = glob.glob(os.path.join(v_root, '*/'))
        v_act_root = sorted(v_act_root)

        # if resume, remember to delete the last video folder
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.*'))
            v_paths = sorted(v_paths)
            v_class = j.split('/')[-2]
            out_dir = os.path.join(out_path, v_class)
            if os.path.exists(out_dir):
                #pass 
                print(out_dir, 'exists!'); continue
            else:
                os.makedirs(out_dir)

            print('extracting: %s' % v_class)
            Parallel(n_jobs=8)(delayed(resize_video_ffmpeg)(vid_list,p, out_path, dim=256) for p in tqdm(v_paths, total=len(v_paths)))



if __name__ == '__main__':
    
    val_vid_gt = '/home/ayushrai/GEBD/data/export/k400_mr345_val_min_change_duration0.3.pkl'
    train_vid_gt = '/home/ayushrai/GEBD/data/export/k400_mr345_train_min_change_duration0.3.pkl'
    test_vid_id_list = '/home/ayushrai/GEBD/data/export/Kinetics400_GEBD_test.txt'

    mode = 'test'  #'test' #val, test

    if mode=='train':
        with open(train_vid_gt, 'rb') as f:
            train_gt_dict = pickle.load(f)
            vid_list = list(train_gt_dict.keys())

    if mode=='val':
        with open(val_vid_gt, 'rb') as f:
            val_gt_dict = pickle.load(f)
            vid_list = list(val_gt_dict.keys())

    if mode=='test':
        test_data = pd.read_csv(test_vid_id_list,header=None)
        vid_list = list(test_data[0])

    #main_kinetics400(vid_list,output_path='/home/ayushrai/kinetics400/GEBD_kinetics400_256')
    main_kinetics400(vid_list,output_path='/media/ayushrai/8TB_HDD/kinetics-dataset/k400_gebd_256')
    # users need to change output_path and v_root