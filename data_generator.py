import os
import random
import skimage
import numpy as np
from collections import namedtuple
from skimage.transform import warp
from scipy.ndimage.interpolation import affine_transform
import nibabel as nib
from nibabel.affines import apply_affine

import transformations as trf
from torchviz import make_dot
from torch.autograd import Variable
import torch

DataObject = namedtuple('DataObject',
                        ('img1', 'img2', 'target_transform', 'img2_orig'))

import torch
from torch.utils import data



class RigidDataLoader(data.Dataset):
    def __init__(self, batch_size, img1_path_pattern, img2_path_pattern, img2_orig_path_pattern, numbers, dataset_count):
        self.batch_size = batch_size
        self.dataset_count = dataset_count
        self._load_images(img1_path_pattern, img2_path_pattern, img2_orig_path_pattern, numbers)        
        # self._create_data_set()

    """
    load all original images and store in RAM
    """
    def _load_images(self, img1_path_pattern, img2_path_pattern, us_orig_path_pattern, numbers):
        # for number in numbers:
        #     img1_path = img1_path_pattern.format(number)
        #     img2_path = img2_path_pattern.format(number)            
        #     img1_nii = nib.load(img1_path)
        #     img2_nii = nib.load(img2_path)
        #     img1 = img1_nii.get_fdata()
        #     img1 = (img1 - np.mean(img1))/(np.max(img1)- np.mean(img1))
        #     img2 = img2_nii.get_fdata()
        #     img2 = (img2 - np.mean(img2))/(np.max(img2)- np.mean(img2))
        #     self.img2_array.append(img1)
        #     self.img1_array.append(img2)

        self.img2_array = []
        self.img1_array = []
        us_orig_list = []
        mr_affine_list = []
        
        for number in numbers:
            mr_path = img1_path_pattern.format(number)
            us_orig_path = us_orig_path_pattern.format(number)
            mr_nii = nib.load(mr_path)
            us_orig_nii = nib.load(us_orig_path)
            us_orig_list.append(us_orig_nii)
            mr_affine_list.append(mr_nii.affine)                        

        max_margins = np.ceil(self._find_cropping_margin(us_orig_list, mr_affine_list)).astype(int)
        self.image_shape = np.copy(max_margins * 2)
        
        for number in numbers:
            mr_path = img1_path_pattern.format(number)
            us_path = img2_path_pattern.format(number)
            us_orig_path = us_orig_path_pattern.format(number)
            mr_nii = nib.load(mr_path)
            us = nib.load(us_path).get_fdata()
            us_orig_nii = nib.load(us_orig_path)
            us_orig_list.append(us_orig_nii)
            mr_affine_list.append(mr_nii.affine)
            mr_cropped, us_cropped = self._crop_to_ultrasound_bounds(mr_nii.get_fdata(), us, us_orig_nii, mr_nii.affine, max_margins)
            mr_cropped = (mr_cropped - np.mean(mr_cropped))/np.std(mr_cropped)
            us_cropped = (us_cropped - np.mean(us_cropped))/np.std(us_cropped)
            self.img1_array.append(mr_cropped)
            self.img2_array.append(us_cropped)

        # self.weight = weight
        # self.SE3_GROUP = SpecialEuclideanGroup(N)
        # self.metric = InvariantMetric( 
        #     group=self.SE3_GROUP, 
        #     inner_product_mat_at_identity=np.eye(SE3_DIM) * self.weight, 
        #     left_or_right='left')

    def _find_cropping_margin(self, us_img_list, mr_affine_list):
        margins = []
        for us_img_orig, mr_affine in zip(us_img_list, mr_affine_list):
            shape = us_img_orig.shape
            pts = np.asarray([[0,0,0], [0,0,shape[2]],[0,shape[1],shape[2]],[0,shape[1],0],
                            [shape[0],0,shape[2]],[shape[0],0,0],[shape[0],shape[1],0],[shape[0],shape[1],shape[2]]])
            pts_wrld = apply_affine(us_img_orig.affine, pts)
            pts_wrld[:,0] -= mr_affine[0,3]
            pts_wrld[:,1] -= mr_affine[1,3]
            pts_wrld[:,2] -= mr_affine[2,3]
            boundries = np.asarray([np.min(pts_wrld, 0), np.max(pts_wrld, 0)]).astype(int)
            center = np.mean(boundries, axis=0)
            margin = boundries[1,:] - center
            margins.append(margin)
        print(np.asarray(margins))
        max_margin = np.asarray(margins).min(axis=0)
        return max_margin


    def _crop_to_ultrasound_bounds(self, mr, us, us_img_orig, mr_affine, max_margins):
        shape = us_img_orig.shape
        pts = np.asarray([[0,0,0], [0,0,shape[2]],[0,shape[1],shape[2]],[0,shape[1],0],
                        [shape[0],0,shape[2]],[shape[0],0,0],[shape[0],shape[1],0],[shape[0],shape[1],shape[2]]])
        pts_wrld = apply_affine(us_img_orig.affine, pts)
        pts_wrld[:,0] -= mr_affine[0,3]
        pts_wrld[:,1] -= mr_affine[1,3]
        pts_wrld[:,2] -= mr_affine[2,3]
        boundries = np.asarray([np.min(pts_wrld, 0), np.max(pts_wrld, 0)]).astype(int)        
        center = np.mean(boundries, axis=0)
        boundries = np.asarray([center - max_margins, center+max_margins]).astype(np.int)
        # print(boundries)
        us_cropped = us[boundries[0,0]:boundries[1,0], boundries[0,1]:boundries[1,1], boundries[0,2]:boundries[1,2]]
        mr_cropped = mr[boundries[0,0]:boundries[1,0], boundries[0,1]:boundries[1,1], boundries[0,2]:boundries[1,2]]
        return mr_cropped, us_cropped

    
    def _get_next_data(self, index):
        img1 = self.img1_array[index]
        img2 = self.img2_array[index]
        base_affine, _ = self._gen_rigid_transform()
        img1 = self._transform_image(img1, base_affine)
        img2 = self._transform_image(img2, base_affine)
        # img2_d, ddf = deform_image(img2, int(10 + 5*(1 - 2*np.random.rand())))            
        affine, se_3 = self._gen_rigid_transform(only_translate=True)
        img2_d = self._transform_image(img2, affine)
        return se_3, self.expand_dim(img2), self.expand_dim(img1), self.expand_dim(img2_d)

    def expand_dim(self, img):
        img = np.expand_dims(img, axis=0)
        return img #np.expand_dims(img, axis=0)

    """
    Generating random rigid transform for data augmentation
    """
    def _gen_rigid_transform(self, only_translate = False):         
        sign_angles = np.where(np.random.rand(3) < 0.5, -1, 1)                  
        angles = np.random.rand(3) * 5 * np.pi / 180 * sign_angles

        if only_translate:
            angles = angles * 0
        max_translate = 5
        sign_translate = np.where(np.random.rand(3) < 0.5, -1, 1)
        translate = (np.random.rand(3) * max_translate * sign_translate).astype(np.int)        
        affine_mat = trf.compose_matrix(angles=angles, translate=translate)
        se_3 = [translate[0]/max_translate, translate[1]/max_translate, translate[2]/max_translate, affine_mat[2,1]-affine_mat[1,2], affine_mat[0,2]-affine_mat[2,0], affine_mat[1,0]-affine_mat[0,1]]
        return affine_mat, se_3

    def _transform_image(self, img, affine_mat):
        tmp_img = np.copy(img)
        return affine_transform(tmp_img, affine_mat)

    """
    Implementing pytorch parallel data loading
    """
    def __getitem__(self, index):
        index = index % len(self.img1_array)
        se_3, img2, img1, img2_d = self._get_next_data(index)
        return DataObject(img1, img2_d, np.asarray(se_3), img2)
    
    def __len__(self):
        # return len(self.img1_array)
        return self.dataset_count


if __name__ == "__main__":
    data_folder = '/home/farid/Documents/eden/dataset/RESECT_RegistrationFiles/images/{0}/'
    # data_folder = '/home/farid/Documents/eden/dataset/corrected/{0}/'
    t1_format = data_folder + 'brainmask_{0}.nii.gz'
    us_format = data_folder + 'Case{0}-US-before_resampled.nii.gz'
    us_format_orig = '/home/farid/Documents/eden/dataset/RESECT/NIFTI/Case{0}/US/Case{0}-US-before.nii.gz'
    training_set = RigidDataLoader(2, t1_format, us_format, us_format_orig, [1,2,3,4,5,6], 1)

    # Generators
    # training_set = Dataset(partition['train'], labels)
    # Parameters
    params = {'batch_size': 2,
            'shuffle': True}
    max_epochs = 5
    training_generator = data.DataLoader(training_set, **params)

    for epoch in range(max_epochs):
        print(epoch)
        # Training
        for idx, (batch_data) in enumerate(training_generator):            
            # Transfer to GPU
            print(len(batch_data))

    # model = UNet3D(10, 10 ,False)
    # X = Variable(torch.randn(2,10,13,14,16))
    # y = model(X)
    # dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    # dot.render("/tmp/model.pdf")
    
    
    # img = img_nii.get_fdata()
    # okno = 20
    # F = b_spline_coeff(okno)
    # X = create_base_grid(img, okno)
    # X_deformed = random_deformation(X, okno)
    # Xx,Xy,Xz = nodes2grid(X_deformed, F, okno, img)
    # warped = warp(img, np.stack((Xy,Xx,Xz)))
    # img_nii_2 = nib.Nifti1Image(warped, img_nii.affine)
    # nib.save(img_nii_2, "/home/farid/Documents/eden/dataset/Lodi_BrainOfTheSheep_first_trip/EXVIVO041619/3DT1-2.nii")