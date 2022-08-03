"""
Script to perfrom detections on point cloud videos
"""

import os
import re
import time
import glob
import cv2
from PIL import Image
import numpy as np
import open3d as o3d

from point_cloud_detect import point_cloud_detect

FPATH = r'test_files\KITTI'
FPATH = r'C:\Users\itber\Documents\datasets\KITTI\kitti_raw_city\2011_09_26_drive_0093_sync\velodyne_points\data'

SAVE_PATH = r'..\point_cloud_images' # don;t track the saved images  (yet)
VIDEO_PATH = os.path.join(SAVE_PATH, '2011_09_26_drive_0093_sync.mp4')
GIF_PATH = os.path.join(SAVE_PATH, '2011_09_26_drive_0093_sync.gif')


def create_gif(save_path, frame_path):
    images = glob.glob(f'{frame_path}/*.png')
    images.sort()

    frames = [Image.open(img) for img in images]

    frames_1 = frames[0]
    frames_1.save(save_path, format='GIF', append_images=frames,
                   save_all=True, duration=85, loop=0)
  
    
# =============================================================================

pcd_paths = glob.glob(os.path.join(FPATH, r'*.pcd'))
save_paths = [os.path.dirname(pcd_paths[0])]

for pcd_path in pcd_paths:
    
    # segment point cloud and get bounding boxes
    pcd = o3d.io.read_point_cloud(pcd_path)
    segmented_pcd, cluster_bboxes = point_cloud_detect(pcd, voxel_size=0.05, 
                        ransac_dist_thresh=0.25, ransac_iters=1000, ransac_n=3, 
                        dbscan_eps=0.5, min_clust=50, max_clust=2500)
    
    # get save path
    # save_path = re.sub(r'\.pcd', '.png', pcd_path)
    
    # new save path
    save_path = os.path.join(SAVE_PATH, 
                             re.sub(r'\.pcd', '.png', 
                                    os.path.basename(pcd_path)))
    save_paths.append(save_path)
    # print(save_path)
    
    # save image
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) 
    _ = [vis.add_geometry(seg) for seg in segmented_pcd]
    _ = [vis.add_geometry(bbox) for bbox in cluster_bboxes]
    _ = [vis.update_geometry(seg) for seg in segmented_pcd]
    _ = [vis.update_geometry(bbox) for bbox in cluster_bboxes]
    
    # get view
    ctr = vis.get_view_control()
    
    ctr.set_lookat(np.array([0., 0., 0.]))
    # ctr.translate(50, 50, -10, -10)
    ctr.rotate(-150., -300., -500., -1500.)
    ctr.rotate(100., 50., -100., 0.)

    ctr.set_zoom(0.1)
    # ctr.change_field_of_view()
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    
    # clean up 
    del ctr
    
    # merge all point clouds into one
    # pcd_combined = o3d.geometry.PointCloud()
    # for seg in segmented_pcd:
    #     pcd_combined += seg
    
    # can't add bounding boxes??
    # for bbox in cluster_bboxes:
    #     pcd_combined += bbox
        
    # save point cloud
    # o3d.io.write_point_cloud(os.path.join(SAVE_PATH, '01.pcd'), pcd_combined)
    
    
# save images to video
img = cv2.imread(save_paths[0])
height, width = img.shape[:2]
height, width = height//2, width//2
video = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'DIVX'), 
                        15, (width,height))

for save_path in save_paths:
    video.write(cv2.resize(cv2.imread(save_path), (width,height), 
                           cv2.INTER_LANCZOS4))

cv2.destroyAllWindows()
video.release()
del video


# create GIF
create_gif(GIF_PATH, SAVE_PATH)

