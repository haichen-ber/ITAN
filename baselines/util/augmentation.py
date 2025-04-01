"""Functions for point cloud data augmentation"""
import numpy as np


###########################################
# numpy based functions
###########################################

def rotate_point_cloud(pc):
    """
    Rotate the point cloud along up direction with certain angle.
    Input:
        pc: Nx3 array of original point clouds
    Return:
        rotated_pc: Nx3 array of point clouds after rotation
    """
    angle = np.random.uniform(0, np.pi * 2)
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    # rotation_matrix = np.array([[cosval, sinval, 0],
    #                             [-sinval, cosval, 0],
    #                             [0, 0, 1]])
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_pc = np.dot(pc, rotation_matrix)

    return rotated_pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter point cloud per point.
    Input:
        pc: Nx3 array of original point clouds
    Return:
        jittered_pc: Nx3 array of point clouds after jitter
    """
    N, C = pc.shape
    assert clip > 0
    jittered_pc = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_pc += pc

    return jittered_pc


def translate_point_cloud(pc):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pc = np.add(np.multiply(pc, xyz1), xyz2).astype('float32')
    return translated_pc

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' pc: Nx3 '''
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def random_scale_point_cloud(data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original point cloud
        Return:
            Nx3 array, scaled point cloud
    """
    N, C = data.shape
    scales = np.random.uniform(scale_low, scale_high)
    data *= scales
    return data

def shift_point_cloud(data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, shifted point cloud
    """
    N, C = data.shape
    data = np.expand_dims(data, axis=0)
    shifts = np.random.uniform(-shift_range, shift_range, (1, 3))
    
    data[0, :, :] += shifts[0, :]
    return data[0]


