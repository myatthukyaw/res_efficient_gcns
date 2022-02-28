import os
import cv2
import json
import torch
import PIL.Image
import argparse
import numpy as np
from PIL import Image

import torch2trt
from torch2trt import TRTModule
import torchvision.transforms as transforms

#import trt_pose.coco
#import trt_pose.models
from . import models, coco
from .draw_objects import DrawObjects
from .parse_objects import ParseObjects


class TrtPose:
    
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.args = args
        
        self.width, self.height = self.args.size, self.args.size
        
        # load humanpose json data
        self.human_pose = self.load_json(args.json)
        
        # load trt model
        self.trt_model  = self._load_trt_model(args.trt_model)
        self.topology = coco.coco_category_to_topology(self.human_pose)
        self.parse_objects = ParseObjects(self.topology)    #, cmap_threshold=0.08, link_threshold=0.08
        self.draw_objects = DrawObjects(self.topology)

        # transformer
        # self.transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
        
        
    @staticmethod
    def load_json(json_file):
        with open(json_file, 'r') as f:
            human_pose = json.load(f)
        return human_pose

    def _load_trt_model(self, MODEL):
        """
        load converted tensorRT model  
        """
        num_parts = len(self.human_pose['keypoints'])
        num_links = len(self.human_pose['skeleton'])

        if MODEL.split('_')[-1] == 'trt.pth':
            model = TRTModule()
            model.load_state_dict(torch.load(MODEL))
            model.eval()
        else:
            if MODEL.split('/')[1].split('_')[0][0:8] == 'densenet':
                model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
            elif MODEL.split('/')[1].split('_')[0][0:6] == 'resnet':
                model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
            model.load_state_dict(torch.load(MODEL))

        return model


    def predict(self, image: np.ndarray):
        """
        predict pose estimationkeypoints
        *Note - image need to be RGB numpy array format
        """
        data = self.preprocess(image)
        cmap, paf = self.trt_model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf) # cmap threhold=0.15, link_threshold=0.15
        return counts, objects, peaks
    
    def preprocess(self, image):
        """
        resize image and transform to tensor image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def draw_objs(self,dst, counts, objects, peaks):
        self.draw_objects(dst, counts, objects, peaks)


def get_keypoint(humans, hnum, peaks):
    kpoint = []
    pid = hnum
    key_points = []
    all_human_kpoints = []
    keypoints = np.zeros((18, 3), dtype=np.float64)
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            # there is a joint : 0
            # peak = peaks[0][j][k]
            # peak = (j, float(peak[1]), float(peak[0]))
            # keypoints[j] = peak

            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:heigh
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            key_points.append(peak[2])
            key_points.append(peak[1])
            # print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            # there is no joint : -1
            # peak = (j, 0., 0.)
            # keypoints[j] = peak

            peak = (j, 0, 0)
            kpoint.append(peak)
            key_points.append(peak[2])
            key_points.append(peak[1])
            # print('index:%d : None'%(j) )
    return pid, key_points#keypoints

openpose_trtpose_match_idx = [
        [0, 0], [14, 2], [15, 1], [16, 4], [17, 3], # head
        [1, 17], [8, 12], [11, 11], # body
        [2, 6], [3, 8], [4, 10], # right hand
        [5, 5], [6, 7], [7, 9], # left hand
        [9, 14], [10, 16], # right leg
        [12, 13], [13, 15] # left leg
    ]

# def trtpose_to_openpose(trtpose_keypoints):
#     """Change trtpose skeleton to openpose format"""
#     #print('trtpose keypoints ', trtpose_keypoints)
#     new_keypoints = trtpose_keypoints.copy()

#     for idx1, idx2 in openpose_trtpose_match_idx:
#         new_keypoints[idx1, 1:] = trtpose_keypoints[idx2, 1:] # neck

#     # for i in range(len(openpose_trtpose_match_idx)):
#     #     openpose_idx = openpose_trtpose_match_idx[i][0] * 2
#     #     trtpose_idx = openpose_trtpose_match_idx[i][1] * 2
#     #     new_keypoints[openpose_idx] = trtpose_keypoints[trtpose_idx]
#     #     new_keypoints[openpose_idx + 1] = trtpose_keypoints[trtpose_idx + 1] 
#     #print('openpose keypoints ',new_keypoints)
#     return new_keypoints

def trtpose_to_openpose(trtpose_keypoints):
    """Change trtpose skeleton to openpose format"""
    #print(trtpose_keypoints)
    new_keypoints = trtpose_keypoints.copy()

    for i in range(len(openpose_trtpose_match_idx)):
        openpose_idx = openpose_trtpose_match_idx[i][0] * 2
        trtpose_idx = openpose_trtpose_match_idx[i][1] * 2
        new_keypoints[openpose_idx] = trtpose_keypoints[trtpose_idx]
        new_keypoints[openpose_idx + 1] = trtpose_keypoints[trtpose_idx + 1] 
    #print('openpose keypoints ',new_keypoints)
    return new_keypoints

def execute(pose, img, dst):
    """
    {people : [ {'person_id':'',pose_keypoints_2d:''} ]}
    """
    # people = {'people':[]}
    # person = {}
    key_points = []
    counts, objects, peaks = pose.predict(img)
    people = []
    for i in range(counts[0]):
        pid, key_points = get_keypoint(objects, i, peaks)
        # person['person_id'] = pid
        # person['pose_keypoints_2d'] = key_points
        # people['people'].append(person)
        # person = {}
        #print(key_points)
        key_points = trtpose_to_openpose(key_points)
        people.append(key_points)
    pose.draw_objs(dst, counts, objects, peaks)
    #print('before', len(people), people)
    #people = remove_persons_with_few_joints(people)
    #print('after', len(people))

    return dst, people, key_points

def remove_persons_with_few_joints(all_keypoints, min_total_joints=10, min_leg_joints=2, include_head=False):
    """Remove bad skeletons before sending to the tracker"""

    good_keypoints = []
    for keypoints in all_keypoints:
        # include head point or not
        total_keypoints = keypoints[5:, 1:] if not include_head else keypoints[:, 1:]
        num_valid_joints = sum(total_keypoints!=0)[0] # number of valid joints
        num_leg_joints = sum(total_keypoints[-7:-1]!=0)[0] # number of joints for legs

        if num_valid_joints >= min_total_joints and num_leg_joints >= min_leg_joints:
            good_keypoints.append(keypoints)
    return np.array(good_keypoints)

def keypoints_to_skeletons_list(all_keypoints):
    ''' Get skeleton data of (x, y * scale_h) from humans.
    Arguments:
        humans {a class returned by self.detect}
        scale_h {float}: scale each skeleton's y coordinate (height) value.
            Default: (image_height / image_widht).
    Returns:
        skeletons {list of list}: a list of skeleton.
            Each skeleton is also a list with a length of 36 (18 joints * 2 coord values).
        scale_h {float}: The resultant height(y coordinate) range.
            The x coordinate is between [0, 1].
            The y coordinate is between [0, scale_h]
    '''
    skeletons_list = []
    NaN = 0
    for keypoints in all_keypoints:
        skeleton = [NaN]*(18*2)
        for idx, kp in enumerate(keypoints):
            skeleton[2*idx] = kp[1]
            skeleton[2*idx+1] = kp[2]
        skeletons_list.append(skeleton)
    return skeletons_list

def expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height):
    """expand bbox for containing more background"""

    width = xmax - xmin
    height = ymax - ymin
    ratio = 0.1   # expand ratio
    new_xmin = np.clip(xmin - ratio * width, 0, img_width)
    new_xmax = np.clip(xmax + ratio * width, 0, img_width)
    new_ymin = np.clip(ymin - ratio * height, 0, img_height)
    new_ymax = np.clip(ymax + ratio * height, 0, img_height)
    new_width = new_xmax - new_xmin
    new_height = new_ymax - new_ymin
    x_center = new_xmin + (new_width/2)
    y_center = new_ymin + (new_height/2)

    return [int(x_center), int(y_center), int(new_width), int(new_height)]


def get_skeletons_bboxes(all_keypoints, image):
    """Get list of (xcenter, ycenter, width, height) bboxes from all persons keypoints"""

    bboxes = []
    img_h, img_w =  image.shape[:2]
    for idx, keypoints in enumerate(all_keypoints):
        keypoints = np.where(keypoints[:, 1:] !=0, keypoints[:, 1:], np.nan)
        keypoints[:, 0] *= img_w
        keypoints[:, 1] *= img_h
        xmin = np.nanmin(keypoints[:,0])
        ymin = np.nanmin(keypoints[:,1])
        xmax = np.nanmax(keypoints[:,0])
        ymax = np.nanmax(keypoints[:,1])
        bbox = expand_bbox(xmin, xmax, ymin, ymax, img_w, img_h)

        # discard bbox with width and height == 0
        if bbox[2] == 0 or bbox[3] == 0:
            continue
        bboxes.append(bbox)

    return bboxes
    

       
        
    
        
                        