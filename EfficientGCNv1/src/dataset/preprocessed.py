import os, pickle, logging, numpy as np
from torch.utils.data import Dataset

from .. import utils as U


class Preprocess_Feeder(Dataset):
    def __init__(self, phase, dataset_path, inputs, num_frame, connect_joint, debug, **kwargs):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        data_path = '{}/{}_data.npy'.format(dataset_path, phase)
        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)
        print(data_path)
        print(label_path)
        try:
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.name, self.label = pickle.load(f, encoding='latin1')
                #self.name, self.label, self.seq_len = pickle.load(f, encoding='latin1')
        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        # if debug:
        #     self.data = self.data[:300]
        #     self.label = self.label[:300]
        #     self.name = self.name[:300]
        #     self.seq_len = self.seq_len[:300]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.name[idx]
        # seq_len = self.seq_len[idx]

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        joint, velocity, bone = self.multi_input(data[:,:self.T,:,:])
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        data_new = np.stack(data_new, axis=0)

        return data_new, label, name

    # def multi_input(self, data):
    #     #print('testing multi input ')
    #     #print(data.shape, data)
    #     C, T, V, M = data.shape
    #     data_new = np.zeros((3, C, T, V, M))
    #     data_new[0,:,:,:,:] = data
    #     for i in range(len(self.conn)):
    #         data_new[1,:,:,i,:] = data[:,:,i,:] - data[:,:,self.conn[i],:]
    #     for i in range(T-1):
    #         data_new[2,:,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
    #     data_new[2,:,T-1,:,:] = 0
    #     return data_new
    
    def multi_input(self, data):
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
