import logging, torch, numpy as np
import model
from tqdm import tqdm
from time import time

#from tensorboardX import SummaryWriter
#from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import utils as U
from . import dataset
from .initializer import Initializer
from .dataset.data_utils import multi_input
from .test import pkummd


def multi_input(data, conn):
    C, T, V, M = data.shape
    data_new = np.zeros((3, C*2, T, V, M))
    data_new[0,:C,:,:,:] = data
    for i in range(V):
        data_new[0,C:,:,i,:] = data[:,:,i,:] - data[:,:,1,:]
    for i in range(T-2):
        data_new[1,:C,i,:,:] = data[:,i+1,:,:] - data[:,i,:,:]
        data_new[1,C:,i,:,:] = data[:,i+2,:,:] - data[:,i,:,:]
    for i in range(len(conn)):
        data_new[2,:C,:,i,:] = data[:,:,i,:] - data[:,:,conn[i],:]
    bone_length = 0
    for i in range(C):
        bone_length += np.power(data_new[2,i,:,:,:], 2)
    bone_length = np.sqrt(bone_length) + 0.0001
    for i in range(C):
        data_new[2,C+i,:,:,:] = np.arccos(data_new[2,i,:,:,:] / bone_length)
    return data_new
    
logging.getLogger().setLevel(logging.INFO)


class Inference(Initializer):

    def extract(self):
        #logging.info('Starting extracting ...')
        #if self.args.debug:
        #    logging.warning('Warning: Using debug setting now!')
        #    logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        #print(U.load_checkpoint(self.args.work_dir, self.model_name))
        #checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        #cm = checkpoint['best_state']['cm']
        #self.model.module.load_state_dict(checkpoint['model'])
        checkpoint = torch.load('pretrained/1001_resgcn-n51-r4_ntu-xsub.pth.tar')
        self.model.module.load_state_dict(checkpoint['model'])

        #
        data = pkummd()

        conn = np.array([9,1,2,3,9,5,6,7,9,9,10,11,10,13,14,15,10,17,18,19]) - 1

        # preprocessing 
        print(data.shape)
        #data = np.expand_dims(data, axis=0)
        data = multi_input(data, conn)
        
        data = np.expand_dims(data, axis=0)
        print(data.shape)
        data = torch.from_numpy(data).float()

        #xx = DataLoader(data, batch_size = 4, shuffle=False, num_workers=1, pin_memory=True)

        # Calculating Output
        self.model.eval()
        out, feature = self.model(data)

        # Processing Data
        data, label = x.numpy(), y.numpy()
        print(out.shape, feature.shape)
        print(out)
        #print(feature)
        predictions = np.argmax(out.cpu().detach().numpy())
        print(predictions)
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.fcn.weight.squeeze().detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()

        print('here ajaja')

        return model


# class Inference(Initializer):

#     def extract(self):
#         #logging.info('Starting extracting ...')
#         #if self.args.debug:
#         #    logging.warning('Warning: Using debug setting now!')
#         #    logging.info('')

#         # Loading Model
#         logging.info('Loading evaluating model ...')
#         #print(U.load_checkpoint(self.args.work_dir, self.model_name))
#         checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
#         cm = checkpoint['best_state']['cm']
#         self.model.module.load_state_dict(checkpoint['model'])

#         # my loading data
#         dataset_name = self.args.dataset.split('-')[0]
#         dataset_args = self.args.dataset_args[dataset_name]
#         self.train_batch_size = dataset_args['train_batch_size']
#         self.eval_batch_size = dataset_args['eval_batch_size']
#         self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
#             self.args.debug, self.args.dataset, **dataset_args
#         )
#         print(self.feeders['eval'])
#         self.train_loader = DataLoader(self.feeders['train'],
#             batch_size=self.train_batch_size, num_workers=4*len(self.args.gpus),
#             pin_memory=True, shuffle=True, drop_last=True
#         )
#         self.eval_loader = DataLoader(self.feeders['eval'],
#             batch_size=self.eval_batch_size, num_workers=4*len(self.args.gpus),
#             pin_memory=True, shuffle=False, drop_last=False
#         )
#         # Loading Data from original extraction
#         x, y, names = iter(self.eval_loader).next()
#         location = self.location_loader.load(names) if self.location_loader else []

#         #
#         data = pkummd()

#         conn = np.array([9,1,2,3,9,5,6,7,9,9,10,11,10,13,14,15,10,17,18,19]) - 1

#         # preprocessing 
#         print(data.shape)
#         #data = np.expand_dims(data, axis=0)
#         data = multi_input(data, conn)
        
#         data = np.expand_dims(data, axis=0)
#         print(data.shape)
#         data = torch.from_numpy(data).float()

#         #xx = DataLoader(data, batch_size = 4, shuffle=False, num_workers=1, pin_memory=True)

#         # Calculating Output
#         self.model.eval()
#         out, feature = self.model(data)

#         # Processing Data
#         data, label = x.numpy(), y.numpy()
#         print(out.shape, feature.shape)
#         print(out)
#         #print(feature)
#         predictions = np.argmax(out.cpu().detach().numpy())
#         print(predictions)
#         out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
#         weight = self.model.module.fcn.weight.squeeze().detach().cpu().numpy()
#         feature = feature.detach().cpu().numpy()

#         print('here ajaja')