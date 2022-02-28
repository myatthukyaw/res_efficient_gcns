import logging, torch, numpy as np
from tqdm import tqdm
from time import time
import argparse
import time
from time import strftime

import os
import yaml
import cv2
from src import utils as U

from torch.utils.data import DataLoader

from utils import  Model,  update_parameters

logging.getLogger().setLevel(logging.INFO)


def init_parameters():
    parser = argparse.ArgumentParser(description='Skeleton-based Action Recognition')

    # input
    # parser.add_argument('--action_model', '-a', help='action recognition model')
    # parser.add_argument('--drop', default=10, help='numbers of frames to drop for each prediction')
    # parser.add_argument('--video_input', help='input video to test')

    # Setting
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Path to pretrained models')
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    parser.add_argument('--no_progress_bar', '-np', default=False, action='store_true', help='Do not show progress bar')
    parser.add_argument('--path', '-p', type=str, default='', help='Path to save preprocessed skeleton files')

    # # Processing
    # parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--visualization', '-v', default=False, action='store_true', help='Visualization')
    parser.add_argument('--generate_data', '-gd', default=False, action='store_true', help='Generate skeleton data')
    parser.add_argument('--generate_label', '-gl', default=False, action='store_true', help='Only generate label')

    # Visualization
    parser.add_argument('--visualization_class', '-vc', type=int, default=0, help='Class: 1 ~ 60, 0 means true class')
    parser.add_argument('--visualization_sample', '-vs', type=int, default=0, help='Sample: 0 ~ batch_size-1')
    parser.add_argument('--visualization_frames', '-vf', type=int, nargs='+', default=[], help='Frame: 0 ~ max_frame-1')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')

    # # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Model type')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')

    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')

    return parser

def get_keypoint(humans, hnum, peaks):
    pid = hnum
    keypoints = np.zeros((18, 3), dtype=np.float64)
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            #there is a joint : 0
            peak = peaks[0][j][k]
            peak = (j, float(peak[1]), float(peak[0]))
            keypoints[j] = peak
        else:    
            # there is no joint : -1
            peak = (j, 0., 0.)
            keypoints[j] = peak
    return pid, keypoints

openpose_trtpose_match_idx = [
        [0, 0], [14, 2], [15, 1], [16, 4], [17, 3], # head
        [1, 17], [8, 12], [11, 11], # body
        [2, 6], [3, 8], [4, 10], # right hand
        [5, 5], [6, 7], [7, 9], # left hand
        [9, 14], [10, 16], # right leg
        [12, 13], [13, 15] # left leg
    ]

def trtpose_to_openpose(trtpose_keypoints):
    """Change trtpose skeleton to openpose format"""
    new_keypoints = trtpose_keypoints.copy()

    for idx1, idx2 in openpose_trtpose_match_idx:
        new_keypoints[idx1, 1:] = trtpose_keypoints[idx2, 1:] # neck

    return new_keypoints


def execute(pose, img, dst):
    key_points = []
    counts, objects, peaks = pose.predict(img)
    people = []
    for i in range(counts[0]):
        pid, key_points = get_keypoint(objects, i, peaks)
        key_points = trtpose_to_openpose(key_points)
        people.append(key_points)
    pose.draw_objs(dst, counts, objects, peaks)

    people = remove_persons_with_few_joints(people)

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

def extract_kpoints(keypoints, fp, l):
    x = keypoints[::2]
    y = keypoints[1::2]
    for n, points in enumerate([x,y]):
        for o, point in enumerate(points):
            fp[n, l, o, 0] = point
    print('l', l,cur_frame, start_frame, end_frame)
    l += 1
    return fp, l

def draw_frame(image, tracks, actions=None, **kwargs):
    """Draw skeleton pose, tracking id and action result on image.
    Check kwargs in func: `draw_trtpose`
    """
    height, width = image.shape[:2]
    thickness = 2 if height*width > (720*960) else 1

    # Draw each of the tracked skeletons and actions text
    for track_id, track in tracks.items():
        #track_id = track['track_id']
        color = (0,255,0)
        #color = get_color_fast(track_id)

        # draw keypoints
        # keypoints = keypoints_list[track['detection_index']]
        # draw_trtpose(image,
        #              keypoints,
        #              thickness=thickness,
        #              line_color=color,
        #              **kwargs)

        # draw track bbox
        x1, y1, x2, y2 = map(int, track['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        print(track_id, actions)
        # draw text over rectangle background
        if track_id in actions.keys():
            label = actions[track_id]
        else:
            label = ''
        # label = actions[track_id] if actions else ''
        label = '{:d}: {}'.format(track_id, label)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.8, thickness)[0]
        yy = (y1 - t_size[1] - 6, y1 - t_size[1] + 14) if y1 - t_size[1] - 5 > 0 \
            else (y1 + t_size[1] + 6, y1 + t_size[1])

        cv2.rectangle(image, (x1, y1), (x1 + t_size[0]+1, yy[0]), color, -1)
        cv2.putText(image, label, (x1, yy[1]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), thickness)


if __name__ == '__main__':

    os.chdir(os.getcwd())

    # Loading Parameters
    parser = init_parameters()
    args, _ = parser.parse_known_args()
    # Updating Parameters (cmd > yaml > default)
    args = update_parameters(parser, args)

    # load pose estimation model
 
    # load tracking model 

    # load action recognition model
    model = Model(args, './workdir')
    action_model = './workdir/2022-01-31 03-39-53/1001_pa-resgcn-b29_zensho.pth.tar'
    model.load(action_model)

    for i in range(50):
        rand_kps = np.random.rand(2,600,18,1)
        model.preprocess(rand_kps)
        actions = model.predict(args.dataset, id, actions)
        print(actions)


    # while(cap.isOpened()):
    #     ret, dst = cap.read()
    #     if ret:

    #         # extract keypoints from each frame using pose estimation model


    #         if bboxes:
    #             # pass skeleton bboxes to deepsort
    #             xywhs = torch.as_tensor(bboxes)
    #             tracks = deepsort.update(xywhs, dst, pair_iou_thresh=0.5)
    #             #print(tracks)

    #             # classify tracked skeletons' actions
    #             if tracks:
    #                 #print(tracks)
    #                 track_keypoints = {track_id: people_list[track['kp_index']]
    #                             for track_id, track in tracks.items()}
    #                 #print(track_keypoints)

    #                 for id, kpoints in track_keypoints.items():

    #                     if not id in frame_track_keypoints.keys():
    #                         print('new id come in', frame_track_keypoints.keys())
    #                         person = dict()
    #                         person['keypoints'] = fp
    #                         person['count'] = 0
    #                         frame_track_keypoints[id] = person
    #                     print(frame_track_keypoints.keys(), frame_track_keypoints[id]['count'])
    #                     x = kpoints[::2]
    #                     y = kpoints[1::2]
    #                     for n, points in enumerate([x,y]):
    #                         for o, point in enumerate(points):
    #                             frame_track_keypoints[id]['keypoints'][n, frame_track_keypoints[id]['count'], o, 0] = point
    #                     #print('l', l,cur_frame, start_frame, end_frame)
    #                     frame_track_keypoints[id]['count'] += 1
    #                     #print(frame_track_keypoints)

    #                     if frame_track_keypoints[id]['count'] == max_frame:
    #                         model.preprocess(frame_track_keypoints[id]['keypoints'])
    #                         actions = model.predict(args.dataset, id, actions)
    #                         #     old_fp = fp.transpose(1,0,2,3)
    #                         #     old_fp = old_fp[drop:max_frame]
    #                         fp = np.zeros((2, max_frame, num_joint, num_person_out), dtype=np.float32)
    #                         person = dict()
    #                         person['keypoints'] = fp
    #                         person['count'] = 0
    #                         frame_track_keypoints[id] = person
            
    #                     draw_frame(dst, tracks, actions)
    #         #model.display_result(dst)
            
    #         # cur_frame += 1
    #         cv2.imshow('a', dst)
    #         c.write(dst)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
        
    # c.destory()