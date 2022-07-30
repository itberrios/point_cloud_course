import os
import open3d as o3d
import numpy as np
import struct

file_to_open = "/Users/jeremycohen/Downloads/Point Clouds/Datasets/data_object_velodyne/testing/velodyne/000020.bin"
file_to_save = "/Users/jeremycohen/Downloads/Point Clouds/test_files/000020.pcd"

def bin2pcd(bin_path):
    ''' 
        Converts a .bin to a .pcd and saves it in it's own directory by 
        default.
        Inputs:
            bin_path (str) abs path to .bin file
            pcd_save_path (str) path to save .pcd file
        Outputs: 
            None
    '''
    
    # get final .pcd save path
    pcd_save_path = os.path.join(os.path.dirname(bin_path),
                                 os.path.basename(bin_path).split('.')[0]
                                 + '.pcd')
    
    size_float = 4
    list_pcd = []
    
    with open (bin_path, "rb") as f:
        byte = f.read(size_float*4)
        while byte:
            x,y,z,intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float*4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)
    
    o3d.io.write_point_cloud(pcd_save_path, pcd)
