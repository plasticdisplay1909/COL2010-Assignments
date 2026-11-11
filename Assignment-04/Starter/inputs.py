"""Provided input helpers. Do not edit. Run setup.py first."""
from pathlib import Path
import json
import numpy as np
from dataset_suite import load_dataset

def available_datasets():
    return [x['name'] for x in json.loads(Path('classification_datasets/manifest.json').read_text())]

def _d():
    with np.load('module4_data.npz',allow_pickle=False) as f:return dict(f)

def get_task1_inputs():d=_d();return d['metric_true'],d['metric_pred'],np.array([0,1])

def get_task2_inputs():d=_d();return d['metric_true'],d['metric_pred']

def get_task3_inputs():d=_d();return d['X_train'], d['X_test']

def get_task4_inputs():d=_d();return d['knn_query'],d['knn_reference'],d['knn_labels'],7,np.array([0,1])

def get_task5_inputs():return (np.array([.05,.2,.5,.8,.95]),)

def get_task6_inputs():d=_d();X=d['gaussian_X'][d['gaussian_y']==1];return d['gaussian_X'][0],X.mean(0),np.maximum(X.var(0),1e-9)

def get_task7_inputs():d=_d();return d['gaussian_X'],d['gaussian_y']

def get_task8_inputs():d=_d();return d['bernoulli_X'],d['bernoulli_y'],1.

def get_task9_inputs():return (np.array([-1000.,-2.25,0.,2.25,1000.]),)

def get_task10_inputs():d=_d();return d['metric_true'],d['probabilities']

def get_task11_inputs():d=_d();return d['X_train_aug'][:240],d['metric_true'],np.array([-.25,-1.,-.7,.9,-.8,1.1]),.1

def get_task12_inputs():X,y,w,l2=get_task11_inputs();return X,w,l2

def get_task13_inputs():d=_d();return d['softmax_Z'],d['softmax_W'],d['softmax_t']

def get_task14_inputs():d=_d();return d['svm_X'],d['svm_y'],d['svm_w'],.2,1.

def get_task15_inputs():d=_d();return d['svm_X'][:30],d['svm_X'][30:40],.8

def get_task16_inputs():return (np.array([17,9,0]),)

def get_task17_inputs():d=_d();return d['tree_feature'][:,None],d['tree_labels']

def get_task18_inputs():return 500,42

def get_task19_inputs():return np.array([[2,5,2,5],[5,5,2,2],[2,2,5,5],[5,5,2,2]]),np.array([2,5])

def get_task20_inputs():X,y,V,t,_,_=load_dataset('imbalanced',standardized=True);p=1/(1+np.exp(-(V.sum(1)-2)));return t,p,np.linspace(.05,.95,19)

def get_task21_inputs():X,y,V,t,_,_=load_dataset('moons',standardized=True);return X,y,V,t,(1,3,5,9,15,31)

def get_task22_inputs():X,y,V,t,_,_=load_dataset('gaussian_independent');return X,y,V,'gaussian',1.

def get_task23_inputs():X,y,_,_,_,_=load_dataset('gaussian_independent',standardized=True);return np.c_[np.ones(len(X)),X],y,.1,50,1e-6

def get_task24_inputs():X,y,V,t,_,_=load_dataset('axis_aligned_rules');return X,y,V,4,2

def get_task25_inputs():X,y,V,t,_,_=load_dataset('axis_aligned_rules');return X,y,V,30,None,None,42
