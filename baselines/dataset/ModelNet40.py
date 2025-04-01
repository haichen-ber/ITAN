import numpy as np
import os
import joblib
import torch
import sys
sys.path.append('../')
sys.path.append('./')
from torch.utils.data import Dataset

from baselines.util.pointnet_utils import normalize_points_np, random_sample_points_np
from baselines.util.augmentation import rotate_point_cloud, jitter_point_cloud


ten_label_indexes = [17, 9, 36, 20, 3, 16, 34, 38, 23]
ten_label_names = ['airplane', 'bed', 'bookshelf', 'bottle', 'chair', 'monitor', 'sofa', 'table', 'toilet', 'vase']

def load_data(data_root, partition='train'):
    npz = np.load(data_root, allow_pickle=True)
    if partition == 'train':
        return npz['train_pc'], npz['train_label']
    elif partition == 'attack':
        return npz['test_pc'], npz['test_label'], npz['target_label']
    elif partition == 'transfer':
        return npz['ori_pc'], npz['test_pc'], npz['test_label']
        # return npz[0], npz[1], npz[2]
    else:
        return npz['test_pc'], npz['test_label']


class ModelNet40(Dataset):
    """General ModelNet40 dataset class."""

    def __init__(self, data_root, num_points, normalize=True,
                 partition='train', augmentation=None):
        assert partition in ['train', 'test']
        self.data, self.label = load_data(data_root, partition=partition)
        self.num_points = num_points
        self.normalize = normalize
        self.partition = partition
        self.augmentation = (partition == 'train') if \
            augmentation is None else augmentation

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3] and its label as a scalar."""
        pc = self.data[item][:, :3]
        if self.partition == 'test':
            pc = pc[:self.num_points]
        else:
            pc = random_sample_points_np(pc, self.num_points)

        label = self.label[item]

        if self.normalize:
            pc = normalize_points_np(pc)

        if self.augmentation:
            pc = rotate_point_cloud(pc)
            pc = jitter_point_cloud(pc)

        return pc, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Hybrid(ModelNet40):
    """ModelNet40 dataset class.
    Add defense point clouds for hybrid training.
    """

    def __init__(self, ori_data, def_data, num_points,
                 normalize=True, partition='train',
                 augmentation=None, subset='ori'):
        assert partition in ['train', 'test']
        ori_data, ori_label = load_data(ori_data, partition=partition)
        ori_data = ori_data[..., :3]
        def_data, def_label = load_data(def_data, partition=partition)
        def_data = def_data[..., :3]
        # concatenate two data
        if partition == 'train':
            self.data = np.concatenate([
                ori_data, def_data], axis=0)
            self.label = np.concatenate([
                ori_label, def_label], axis=0)
        else:  # only take subset data for testing
            if subset == 'ori':
                self.data = ori_data
                self.label = ori_label
            elif subset == 'def':
                self.data = def_data
                self.label = def_label
            else:
                print('Subset not recognized!')
                exit(-1)
        # shuffle real and defense data
        if partition == 'train':
            idx = list(range(len(self.label)))
            np.random.shuffle(idx)
            self.data = self.data[idx]
            self.label = self.label[idx]
        self.num_points = num_points
        self.normalize = normalize
        self.partition = partition
        self.augmentation = (partition == 'train') if \
            augmentation is None else augmentation


class ModelNet40Normal(Dataset):
    """Modelnet40 dataset with point normals.
    This is used in kNN attack which requires normal in projection operation.
    """

    def __init__(self, data_root, num_points, normalize=True):
        self.data, self.label = \
            load_data(data_root, partition='test')
        self.num_points = num_points
        # not for training, so no need to consider augmentation
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 6] and its label as a scalar."""
        pc = self.data[item][:self.num_points, :6]
        label = self.label[item]

        if self.normalize:
            pc[:, :3] = normalize_points_np(pc[:, :3])

        return pc, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40Attack(Dataset):
    """Modelnet40 dataset for target attack evaluation.
    We return an additional target label for an example.
    """

    def __init__(self, data_root, num_points, normalize=True):
        self.data, self.label, self.target = \
            load_data(data_root, partition='attack')
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3], its label as a scalar
            and its target label for attack as a scalar.
        """
        pc = self.data[item][:self.num_points, :3]
        label = self.label[item]
        target = self.target[item]

        if self.normalize:
            pc = normalize_points_np(pc)

        return pc, label, target

    def __len__(self):
        return self.data.shape[0]


class ModelNet40NormalAttack(Dataset):
    """Modelnet40 dataset with point normals and target label."""

    def __init__(self, data_root, num_points, normalize=True):
        self.data, self.label, self.target = \
            load_data(data_root, partition='attack')
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 6], its label as a scalar
            and its target label for attack as a scalar.
        """
        pc = self.data[item][:self.num_points, :6]
        label = self.label[item]
        target = self.target[item]

        if self.normalize:
            pc[:, :3] = normalize_points_np(pc[:, :3])

        return pc, label, target

    def __len__(self):
        return self.data.shape[0]


class CustomModelNet40(Dataset):
    """Modelnet40 dataset for target attack evaluation.
    We return an additional target label for an example.
    """

    def __init__(self, data_root, num_points, normalize=True):
        data_all=joblib.load(os.path.join(data_root,'attacked_data.z'))
        self.data = np.concatenate(data_all, axis=0)
        print(self.data.shape)
        label_lst = []
        for i in range(40):
            lst = [i] * data_all[i].shape[0]
            label_lst = label_lst + lst
        #print(label_lst)
        self.label = np.array(label_lst)
        self.num_points = num_points
        self.normalize = normalize

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3], its label as a scalar
            and its target label for attack as a scalar.
        """
        pc = self.data[item][:self.num_points, :3]
        label = self.label[item]

        if self.normalize:
            pc = normalize_points_np(pc)

        return pc, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40_Geo():
    def __init__(self, data_mat_file='../Data/modelnet10_250instances_1024.mat', attack_label='All', resample_num=-1,
                 is_half_forward=False):
        self.data_root = data_mat_file
        self.attack_label = attack_label
        self.is_half_forward = is_half_forward

        if not os.path.isfile(self.data_root):
            assert False, 'No exists .npz file!'

        dataset = load_data(self.data_root, partition='attack')
        # data = torch.FloatTensor(dataset['data'])
        # normal = torch.FloatTensor(dataset['normal'])
        # label = dataset['label']
        data = torch.FloatTensor(dataset[0])[:, :, :3]
        normal = data
        label = dataset[1][:, np.newaxis]
        if resample_num > 0:
            tmp_data_set = []
            tmp_normal_set = []
            for j in range(data.size(0)):
                tmp_data, tmp_normal = self.__farthest_points_normalized(data[j].t(), resample_num, normal[j].t())
                tmp_data_set.append(torch.from_numpy(tmp_data).t().float())
                tmp_normal_set.append(torch.from_numpy(tmp_normal).t().float())
            data = torch.stack(tmp_data_set)
            normal = torch.stack(tmp_normal_set)

        if attack_label in ten_label_names:
            for k, label_name in enumerate(ten_label_names):
                if attack_label == label_name:
                    self.start_index = k * 25
                    self.data = data[k * 25:(k + 1) * 25]
                    self.normal = normal[k * 25:(k + 1) * 25]
                    self.label = label[k * 25:(k + 1) * 25]
        elif attack_label == 'All':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        elif attack_label == 'Untarget' or attack_label == 'Random':
            self.start_index = 0
            self.data = data
            self.normal = normal
            self.label = label
        else:
            assert False

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):

        if (self.attack_label in ten_label_names) or (self.attack_label == 'All'):
            label = self.label[index]

            target_labels = []
            for i in ten_label_indexes:
                if label != i:
                    target_labels.append(i)
                else:
                    target_labels.append(i + 1)
            target_labels = torch.IntTensor(np.array(target_labels)).long()
            gt_labels = torch.IntTensor(label).long().expand_as(target_labels)
            assert target_labels.size(0) == 9

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(9, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(9, -1, -1)
            if self.is_half_forward:
                return [[pcs[:4, :, :], normals[:4, :, :], gt_labels[:4], target_labels[:4]],
                        [pcs[4:, :, :], normals[4:, :, :], gt_labels[4:], target_labels[4:]]]
            else:
                return [pcs, normals, gt_labels, target_labels]

        elif (self.attack_label == 'Untarget'):
            label = self.label[index]
            gt_labels = torch.IntTensor(label).long()

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(1, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(1, -1, -1)
            return [pcs, normals, gt_labels]

        elif (self.attack_label == 'Random'):
            label = self.label[index]
            gt_labels = torch.IntTensor(label).long()

            pc = self.data[index].contiguous().t()
            pcs = pc.unsqueeze(0).expand(1, -1, -1)

            normal = self.normal[index].contiguous().t()
            normals = normal.unsqueeze(0).expand(1, -1, -1)

            target_labels = torch.IntTensor([choice([i for i in range(0, 40) if i not in [gt_labels.item()]])]).long()

            return [pcs, normals, gt_labels, target_labels]

    def __farthest_points_normalized(self, obj_points, num_points, normal):
        first = np.random.randint(len(obj_points))
        selected = [first]
        dists = np.full(shape=len(obj_points), fill_value=np.inf)

        for _ in range(num_points - 1):
            dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis=1))
            selected.append(np.argmax(dists))
        res_points = np.array(obj_points[selected])
        res_normal = np.array(normal[selected])

        # normalize the points and faces
        avg = np.average(res_points, axis=0)
        res_points = res_points - avg[np.newaxis, :]
        dists = np.max(np.linalg.norm(res_points, axis=1), axis=0)
        res_points = res_points / dists

        return res_points, res_normal
    

class ModelNet40Transfer(Dataset):
    """Modelnet40 dataset for target attack evaluation.
    We return an additional target label for an example.
    """

    def __init__(self, data_root, num_points):
        self.ori_data, self.adv_data, self.label = \
            load_data(data_root, partition='transfer')
        self.num_points = num_points

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3], its label as a scalar
            and its target label for attack as a scalar.
        """
        ori_pc = self.ori_data[item][:self.num_points, :3]
        if self.adv_data.shape[1]==3:
            self.adv_data = self.adv_data.transpose((0, 2, 1))
        adv_pc = self.adv_data[item][:self.num_points, :3]
        label = self.label[item]

        ori_pc = normalize_points_np(ori_pc)


        return ori_pc, adv_pc, label

    def __len__(self):
        return self.adv_data.shape[0]