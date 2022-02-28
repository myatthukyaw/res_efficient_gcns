import os
import yaml
from time import time
from time import strftime
import logging, torch, numpy as np

from src.initializer import Initializer

def update_parameters(parser, args):
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

class Model(Initializer):
    def __init__(self, args, save_dir) -> None:
        super().__init__(args, save_dir)
        self.args = args
        self.conn = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
        self.top_p = 0
        self.my_dataset_action_names = ['action1', 'action2', 'action3']

    def load(self, model):
        # Loading Model
        logging.info('Loading evaluating model ...')       
        checkpoint = torch.load(model)
        self.model.module.load_state_dict(checkpoint['model'])
        self.model.eval()

    def multi_input(self, data) -> list:
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
        self.data = self.multi_input(fp)
        self.data = np.expand_dims(self.data, axis=0)
        self.data = torch.from_numpy(self.data).float()

    def predict(self) -> str:
        out, _ = self.model(self.data)
        prob = torch.nn.functional.softmax(out, dim=1)
        self.top_p, _ = prob.topk(1, dim = 1)
        prediction = np.argmax(out.cpu().detach().numpy())
        self.action_predicted = self.my_dataset_action_names[prediction]
        action = self.action_predicted + '-'+ str(round(100 * self.top_p[0][0].cpu().detach().numpy())) + '%'
        return action