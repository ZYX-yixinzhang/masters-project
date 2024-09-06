"""
    This file includes the preprocessing steps for image data,
    including:
        read data from directory
        reorientate
        bias-field correction
        resampling/clipping
        and store the processed items
"""

import os
import ants
from helper import *

path = '../dataset/p2/ADNI'
def read_pics(path):
    fold = os.listdir(path)
    mris = {}
    for n in fold:
        f = os.listdir(path+'/'+n)
        key = n
        for j in f:
            f2 = os.listdir(path+'/'+n+'/'+j)
            for d in f2:
                f3 = os.listdir(path+'/'+n+'/'+j+'/'+d)
                for pic in f3:
                    pic_folder = os.listdir(path+'/'+n+'/'+j+'/'+d+'/'+pic)
                    for pict in pic_folder:
                        mris[key] = path+'/'+n+'/'+j+'/'+d+'/'+pic+'/'+pict
    return mris
mris = read_pics(path)

class preprocess():
    def __init__(self, template, template_mask):
        self.template = ants.image_read(template, reorient='IAL')
        self.template_mask = ants.image_read(template_mask, reorient='IAL')
        
    def bias_field_correction(self, img):
        corrected = ants.n4_bias_field_correction(img)
        return corrected

    def Brain_extraction(self, img):
        brain_mask = ants.apply_transforms(
            fixed=self.template['warpedmovout'],
            moving=template_img_ants_mask,
            transformlist=self.template['fwdtransforms'],
            interpolator='nearestNeighbor',
            verbose=True
        )
        brain_mask_dilated = ants.morphology(brain_mask, radius=4, operation='dilate', mtype='binary')
        masked = ants.mask_image(img, brain_mask_dilated)
        return masked

    def resample(self, img, size):
        resampled = ants.resample_image(img, size, True, 2)
        return resampled
    
minimum1 = 192
maximum1 = 256
minimum2 = 192
maximum2 = 256
minimum3 = 160
maximum3 = 196
s1, t1 = 100, 200
s2, t2 = 100, 200
s3, t3 = 51, 151
def clip(d1, d2, d3):
    return [int(d1*s1/maximum1-50*(1-d1/maximum1)), int(d1*t1/maximum1+50*(1-d1/maximum1)), int(d2*s2/maximum2-50*(1-d2/maximum2)), int(d2*t2/maximum2+50*(1-d2/maximum2)), int(d3*s3/maximum3-50*(1-d3/maximum3)), int(d3*t3/maximum3+50*(1-d3/maximum3))]

def pipline(img, processor, bias=False, mask=False, resample=False):
    if mask:
        preprocessor.template = ants.registration(
            fixed=img,
            moving=preprocessor.template, 
            type_of_transform='SyN',
            verbose=True
        )
        img = preprocessor.Brain_extraction(img)
    if bias:
        # print('biased')
        img = preprocessor.bias_field_correction(img)
    if resample:
        img = preprocessor.resample(img, (110, 110, 110))
    return img.numpy()
    
import ants
from helper import *
template_img_path_mask = './dataset/template/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii'
template_img_path = './dataset/template/mni_icbm152_t1_tal_nlin_sym_09a.nii'
preprocessor = preprocess(template_img_path, template_img_path_mask)

def proc_pic(processor, mris, bias=False, mask=False, resample=False):
    procesed = {}
    for key, item in mris.items():
        p = ants.image_read(item, reorient='IAL')
        procesed[key]=pipline(p, processor, bias, mask, resample)
        [d1, d2, d3] = procesed[key].shape
        a1, b1, a2, b2, a3, b3 = clip(d1, d2, d3)
        procesed[key]=procesed[key][a1:b1, a2:b2, a3:b3]
        print('processed the patient ', key, procesed[key].shape)
    return procesed
print(len(mris))
procesed = proc_pic(preprocessor, mris, True, False, False)

for key, item in procesed.items():
    print(key)
    with open('./procesed/n/'+key+'.txt', 'w') as outfile:
        
        for slice_2d in item:
            np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')