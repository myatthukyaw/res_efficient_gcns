import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .preprocessor import pre_normalization

classes = {
    'action 1' : 0,
    'action 2' : 1,
    'action 3' : 2,
}

class MyDataset_Generator():
    def __init__(self, args, dataset_args):
        self.dataset_args = dataset_args
        self.num_person_out = 1
        self.num_person_in = 4
        self.num_joint = 18
        self.max_frame = 700
        self.dataset = args.dataset
        self.print_bar = not args.no_progress_bar
        self.generate_label = args.generate_label

        self.out_path = '{}/{}'.format(dataset_args['path'], self.dataset.replace('-', '/'))
        U.create_folder(self.out_path)


    def start(self):
        # Generate data
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))

            file_list = []
            folder = self.dataset_args['my_dataset_data_path']
            phase_folder = os.path.join(folder, phase)
            for filename in os.listdir(phase_folder):
                file_list.append((phase_folder, filename))
            self.gendata(phase, file_list)


    def read_xyz(self, file):
        # seq_info = self.read_skeleton_filter(file)
        seq_info = np.load(file)
        data = np.zeros((self.num_person_out, seq_info.shape[0], self.num_joint, 2))
        for n, f in enumerate(seq_info):
            for i in range(self.num_joint):
                data[0, n, i, :] = [f[i*2], f[i*2+1]]

        data = data.transpose(3, 1, 2, 0)  # to (C,T,V,M)
        return data


    def gendata(self, phase, file_list):
        sample_name = []
        sample_label = []
        sample_paths = []
        for folder, filename in sorted(file_list):

            path = os.path.join(folder, filename)
            action_class = filename.split('_')[0]

            if action_class in classes.keys():
                action_class = classes[action_class]
            else:
                print("{} not found".format(action_class))
                continue

            sample_paths.append(path)
            sample_label.append(action_class)  # to 0-indexed
        #print(len(sample_label), len(sample_paths))

        # Save labels
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_paths, list(sample_label)), f)

        if not self.generate_label:
            # Create data tensor (N,C,T,V,M)
            fp = np.zeros((len(sample_label), 2, self.max_frame, self.num_joint, self.num_person_out), dtype=np.float32)

            # Fill (C,T,V,M) to data tensor (N,C,T,V,M)
            items = tqdm(sample_paths, dynamic_ncols=True) if self.print_bar else sample_paths
            for i, s in enumerate(items):
                data = self.read_xyz(s)
                fp[i, :, 0:data.shape[1], :, :] = data   

            # Perform preprocessing on data tensor
            fp = pre_normalization(fp, print_bar=self.print_bar)

            # Save input data (train/eval)
            print(fp.shape)
            np.save('{}/{}_data.npy'.format(self.out_path, phase), fp)
