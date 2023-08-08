import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='shapenetcorev2', num_points=2048, split='train', load_name=True,
                 load_file=True, ):

        #self.root = os.path.join(root, dataset_name + '_' + 'hdf5_2048')
        self.root = root
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file

        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train', 'trainval', 'all']:
            self.get_path('train')
        if self.split in ['val', 'trainval', 'all']:
            self.get_path('val')
        if self.split in ['test', 'all']:
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)

        if self.load_name:
            self.path_name_all.sort()
            self.name = self.load_json(self.path_name_all)  # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = self.load_json(self.path_file_all)  # load file name
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '%s*.h5' % type)
        print(path_h5py)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json' % type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.root, '%s*_id2file.json' % type)
            self.path_file_all += glob(path_json)
        return

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)

        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = self.label[item]
        return point_set, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    rootdir = './PointCloudDatasets'
    dataset_name = 'shapenetcorev2'
    # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
    # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
    split = 'train'
    d = Dataset(root=rootdir, dataset_name=dataset_name, num_points=2048, split=split)
    pc = d[0][0]
    print('min:', pc.min())
    print('max:', pc.max())
