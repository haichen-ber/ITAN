import numpy as np
import argparse
import os
from dataset import ModelNet40Attack, ModelNet40Transfer, load_data

def merge(args):
    data_root = args.data_root
    f1 = args.f1
    f2 = args.f2
    ori_data_lst = []
    adv_data_lst = []
    label_lst = []
    save_name = data_root+"-concat.npz"
    if os.path.exists(os.path.join(data_root, save_name)):
        return os.path.join(data_root, save_name)
    
    # f1
    file_path = os.path.join(data_root, f1)
    ori_data, adv_data, label = \
        load_data(file_path, partition='attack')
    ori_data_lst.append(ori_data)
    adv_data_lst.append(adv_data)
    label_lst.append(label)

    # f2
    file_path = os.path.join(data_root, f2)
    ori_data, adv_data, label = \
        load_data(file_path, partition='attack')
    ori_data_lst.append(ori_data)
    adv_data_lst.append(adv_data)
    label_lst.append(label)

    all_ori_pc = np.concatenate(ori_data_lst, axis=0)  # [num_data, K, 3]
    all_adv_pc = np.concatenate(adv_data_lst, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(label_lst, axis=0)  # [num_data]

    np.savez(os.path.join(data_root, save_name),
             test_pc=all_ori_pc.astype(np.float32),
             test_label=all_adv_pc.astype(np.float32),
             target_label=all_real_lbl.astype(np.uint8))
    return os.path.join(data_root, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='')
    parser.add_argument('--f1', type=str, default='normal',
                        help='file 1')
    parser.add_argument('--f2', type=str, default='normal',
                        help='file 2')
    args = parser.parse_args()
    datan = merge(args)