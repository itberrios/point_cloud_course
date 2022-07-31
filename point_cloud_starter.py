'''
@article{Zhou2018,
	author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
	title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
	journal   = {arXiv:1801.09847},
	year      = {2018},
}
'''
## IMPORT LIBRARIES
import numpy as np
import time
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt

from point_cloud_utils import get_clusters

## USE http://www.open3d.org/docs/release/tutorial/Basic/

## 1 - open file and visualize the point cloud
# The supported extension names are: pcd, ply, xyz, xyzrgb, xyzn, pts.
# pcd_path = r'test_files\fragment.ply'
pcd_path = r'test_files\sdc.pcd'
pcd = o3d.io.read_point_cloud(pcd_path)

## IF YOU HAVE PPTK INSTALLED, VISUALIZE USING PPTK
#import pptk
#v = pptk.viewer(pcd.points)

## 2 perform Voxel Grid Downsampling
print(f"Points before downsampling: {len(pcd.points)} ")
print("Downsample the point cloud with a voxel of 0.05")
pcd = pcd.voxel_down_sample(voxel_size=0.05)
# o3d.visualization.draw_geometries([pcd])
                                  
print(f"Points after downsampling: {len(pcd.points)}")# DOWNSAMPLING


## 3 Segement ground plane from the rest of the point cloud
tic = time.time()
_, inliers = pcd.segment_plane(distance_threshold=0.25,
                               ransac_n=3,
                               num_iterations=1000)
toc = time.time()
print(f'Time to segment image with RANSAC: {toc - tic}')

# visualize segmented point cloud
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0,1,1])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([1,0,1])

# o3d.visualization.draw_geometries([inlier_cloud , outlier_cloud])
# o3d.visualization.draw_geometries([outlier_cloud])


## 4 - CLUSTERING USING DBSCAN
outlier_cloud.paint_uniform_color([0,0,0])
inlier_cloud.paint_uniform_color([0,0,0])

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.5, 
                                                   min_points=12, 
                                                   print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# display clustered point cloud
colors = plt.get_cmap('tab20')(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# display indiviudal clusters
cluster_clouds = get_clusters(outlier_cloud, colors, background_color=[0,0,0])
o3d.visualization.draw_geometries(cluster_clouds)
    
    
    
'''

## BONUS CHALLENGE - CLUSTERING USING KDTREE AND KNN INSTEAD
# http://www.open3d.org/docs/latest/tutorial/Basic/kdtree.html
pcd_tree = 

## CHALLENGE 5 - BOUNDING BOXES IN 3D
bounding_boxes = 

## CHALLENGE 6 - VISUALIZE THE FINAL RESULTS
list_of_visuals = 

## BONUS CHALLENGE 2 - MAKE IT WORK ON A VIDEO

'''
