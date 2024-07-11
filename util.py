import numpy as np


def patchify_voxel(volume, patch_size, pad_size):
    """
    volume: 3D numpy array
    patch_size: tuple of patch size (p_h, p_w, p_d)
    """

    p_h, p_w, p_d = patch_size
    
    v_h, v_w, v_d = volume.shape

    pad_h, pad_w, pad_d = pad_size

    # Calculate the number of patch in ach axis
    n_w = np.ceil(1.0*(v_w-p_w)/p_w+1)
    n_h = np.ceil(1.0*(v_h-p_h)/p_h+1)
    n_d = np.ceil(1.0*(v_d-p_d)/p_d+1)

    n_w = int(n_w)
    n_h = int(n_h)
    n_d = int(n_d)

    pad_1 = (n_w - 1) * p_w + p_w - v_w
    pad_2 = (n_h - 1) * p_h + p_h - v_h
    pad_3 = (n_d - 1) * p_d + p_d - v_d

    volume = np.pad(volume, ((0, pad_1), (0, pad_2), (0, pad_3)), mode='reflect')
    
    h, w, d= volume.shape
    x_ = np.int32(np.linspace(0, h-p_h, n_h))
    y_ = np.int32(np.linspace(0, w-p_w, n_w))
    z_ = np.int32(np.linspace(0, d-p_d, n_d))
    
    ind = np.meshgrid(x_, y_, z_, indexing='ij')
    
    patch_list = []
    start_ind = []

    volume = np.pad(volume, ((pad_h, pad_h), (pad_w, pad_w), (pad_d, pad_d)), mode='reflect')

    for i, start in enumerate(list(np.array(ind).reshape(3,-1).T)):
        patch = volume[start[0]:start[0]+p_h+2*pad_h, start[1]:start[1]+p_w+2*pad_w, start[2]:start[2]+p_d+2*pad_d]
        patch_list.append(patch)
        start_ind.append(start)
        
    return patch_list, start_ind, (h, w, d), (v_h, v_w, v_d)


def unpatchify_voxel(patch_list, start_ind, patch_size, imsize=(128,128,128), original_shape=(128,128,128)):
    """
    patch_list: list of patches
    start_ind: list of starting index of each patch
    seq_ind: list of sequence index of each patch
    imsize: shape of the image
    original_shape: original shape of the image
    """
    p_h, p_w, p_d = patch_size

    volume = np.zeros(imsize)
    for i, (patch, start) in enumerate(zip(patch_list, start_ind)):
        volume[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d] = patch
        
    # crop the padded region
    volume = volume[:original_shape[0], :original_shape[1], :original_shape[2]]

    return volume