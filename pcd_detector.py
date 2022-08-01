"""
Script to perfrom detections on point cloud videos
"""

import os
import re
import time
import glob
import tqdm
import numpy as np
import open3d as o3d

from point_cloud_detect import point_cloud_detect

FPATH = r'test_files\KITTI'

pcd_paths = glob.glob(os.path.join(FPATH, r'*.pcd'))
save_paths = [os.path.dirname(pcd_paths[0])]

for pcd_path in pcd_paths:
    
    # get bounding boxes
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    segmented_pcd, cluster_bboxes = point_cloud_detect(pcd, voxel_size=0.05, 
                       ransac_dist_thresh=0.25, ransac_iters=500, ransac_n=3, 
                       dbscan_eps=0.5, min_clust=50, max_clust=2500)
    
    # get save path
    save_path = re.sub(r'\.pcd', '.png', pcd_path)
    
    # save image
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) #works for me with False, on some systems needs to be true
    _ = [vis.add_geometry(seg) for seg in segmented_pcd]
    _ = [vis.add_geometry(bbox) for bbox in cluster_bboxes]
    _ = [vis.update_geometry(seg) for seg in segmented_pcd]
    _ = [vis.update_geometry(bbox) for bbox in cluster_bboxes]
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    
    
    