import os
import yaml
import cv2
import logging, torch, numpy as np
from tqdm import tqdm
from time import time
import argparse
import time
from time import strftime

from src.initializer import Initializer


# def get_save_dir(args):
#     if args.debug or args.evaluate or args.extract or args.visualization or args.generate_data or args.generate_label:
#         save_dir = '{}/temp'.format(args.work_dir)
#     else:
#         ct = strftime('%Y-%m-%d %H-%M-%S')
#         save_dir = '{}/{}_{}_{}/{}'.format(args.work_dir, args.config, args.model_type, args.dataset, ct)
#     U.create_folder(save_dir)
#     return save_dir

def update_parameters(parser, args):
    #print(os.path.exists('./configs/{}.yaml'.format(args.config)))
    if os.path.exists('./configs/{}.yaml'.format(args.config)):
        with open('./configs/{}.yaml'.format(args.config), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    return parser.parse_args()

def add_white_region_to_left_of_image(img_disp):
    r, c, d = img_disp.shape
    blank = 255 + np.zeros((int(r/4),c , d), np.uint8)
    img_disp = np.vstack((img_disp, blank))
    return img_disp

class Camera():
    def start(self, test_video):
        self.cap = cv2.VideoCapture(test_video)
        self.video_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.out_video = cv2.VideoWriter(os.path.join('outputs','output.mp4'), self.fourcc, self.cap.get(cv2.CAP_PROP_FPS), (1080,900))
        return self.cap

    def write(self, dst):
        self.out_video.write(dst)

    def destory(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.out_video.release()

class Model(Initializer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.conn = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
        self.top_p = 0
        self.gw_action_names = [
             'fighting', 'smoke', 'stand', 'walk'
        ]
        self.rtar_action_names = [
             'jump','kick', 'punch', 'sit','squat', 'stand', 'walk','wave'
        ]
        self.zensho_action_names = ['Input', 'Remove', 'Submerge']

    def load(self, action_model):
        # Loading Model
        logging.info('Loading evaluating model ...')
        if self.args.dataset == 'rtar':
            model = 'workdir/2001_EfficientGCN-B4_rtar/2021-08-04 13-24-15/2001_EfficientGCN-B4_rtar.pth.tar'
        elif self.args.dataset == 'gw':
            model = os.path.join('workdir/1001_pa-resgcn-b29_gw', action_model, '1001_pa-resgcn-b29_gw.pth.tar')
        elif self.args.dataset == 'zensho':
            model = os.path.join('workdir/1001_pa-resgcn-b29_zensho/2022-01-31 03-39-53/1001_pa-resgcn-b29_zensho.pth.tar')
        checkpoint = torch.load(model)
        self.model.module.load_state_dict(checkpoint['model'])
        self.model.eval()


    def multi_input(self, data, conn):
        C, T, V, M = data.shape
        data_new = np.zeros((3, C*2, T, V, M))
        data_new[0,:C,:,:,:] = data
        for i in range(V):
            data_new[0,C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
        for i in range(T-2):
            data_new[1,:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
            data_new[1,C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
        for i in range(len(self.conn)):
            data_new[2,:C,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
        bone_length = 0
        for i in range(C):
            bone_length += np.power(data_new[2,i,:,:,:], 2)
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            data_new[2,C+i,:,:,:] = np.arccos(data_new[2,i,:,:,:] / bone_length)
        return data_new

    def preprocess(self, fp):
        self.data = self.multi_input(fp, self.conn)
        print(self.data.shape)
        self.data = np.expand_dims(self.data, axis=0)
        self.data = torch.from_numpy(self.data).float()

    def predict(self, dataset, id, actions):
        out, feature = self.model(self.data)
        prob = torch.nn.functional.softmax(out, dim=1)
        self.top_p, top_class = prob.topk(1, dim = 1)
        prediction = np.argmax(out.cpu().detach().numpy())
        if dataset == 'gw':
            self.action_predicted = self.gw_action_names[prediction]
        elif dataset == 'rtar':
            self.action_predicted = self.rtar_action_names[prediction]
        print('predicted action ', self.action_predicted)
        actions[id] = self.action_predicted + '-'+ str(round(100 * self.top_p[0][0].cpu().detach().numpy())) + '%'
        return actions
    
    def display_result(self, dst):
        if self.top_p >0.8:
            dst = cv2.putText(
                dst, 
                "Prediction : " + str(self.action_predicted) + "( "+ str(round(100 * self.top_p[0][0].cpu().detach().numpy())) +"% prob)" , 
                (100, 850), 
                fontScale=1.5, 
                fontFace=cv2.FONT_HERSHEY_PLAIN, 
                color=(0,0,255), 
                thickness=2)