"""
Script to detect objects in a point cloud
The supported extension names are: pcd, ply, xyz, xyzrgb, xyzn, pts.

usages:
    python point_cloud_detect --pcd_path path/to/file.pcd

This script uses unsupervised learning to segment the point cloud and find
objects of interest. It does this in 6 main steps
    1) Perform Voxel Downsampling to decrease the total number of points
    2) Perform planar segmentation via RANSAC to determine the surface VS all 
           other parts of the point cloud
    3) Perform DBSCAN clustering to find objects of interest
    4) Use PCA to find the axial orienation of each clustered object 
           (Axial orientation relative to the origin 
            i.e. 1st principle component)
    5) Obtain bounding boxes for each clustered object with correct axis
           alignment obtained in step 4
           (open3d handles steps 4 and 5 with 'get_axis_aligned_bbox(pcd)')
    6) Place everything in a list to display with open3d
       
This is a relatively slow process in Python, 5-8 Hz is a reasonable expectation
for most point clouds given that the voxel size is large enough. For a more 
realistic implementation the coloring can be dropped to increase the speed.
"""
import argparse
import time
import numpy as np
import open3d as o3d

from point_cloud_utils import get_clusters_from_labels, \
                              get_axis_aligned_bbox

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('--pcd_path', type=str, help='Path to .pcd file')
ap.add_argument('--voxel_size', type=float, default=0.1, 
                help='Voxel Size for down sampling')
ap.add_argument('--ransac_dist_thresh', type=float, default=0.25,
                help='RANSAC segmentation distance threshold')
ap.add_argument('--ransac_iters', type=int, default=500,
                help='RANSAC segmentation iterations')
ap.add_argument('--ransac_n', type=int, default=3,
                help='min points to define a plane, best left at 3 or 4')
ap.add_argument('--dbscan_eps', type=float, default=0.5,
                help='DBSCAN cluster neighborhood distance')
ap.add_argument('--min_clust', type=int, default=30,
                help='minimum points required to declare a cluster')
ap.add_argument('--max_clust', type=int, default=2000,
                help='mmaximum points to declare a cluster')


args = vars(ap.parse_args())

# 
# 0 
# pcd_path = r'test_files\sdc.pcd'
# # 1 - voxel downsampling
# voxel_size = 0.15
# # 2 - RANSAC segmentation
# ransac_dist_thresh = 0.25
# ransac_iters = 500
# ransac_n = 3
# # 3 DBSCAN 
# dbscan_eps = 0.5
# min_clust = 30

# # 6 visualize
# max_clust = 2000

pcd_path = args['pcd_path']
voxel_size = args['voxel_size']
ransac_dist_thresh = args['ransac_dist_thresh']
ransac_iters = args['ransac_iters']
ransac_n = args['ransac_n']
dbscan_eps = args['dbscan_eps']
min_clust = args['min_clust']
max_clust = args['max_clust']

tic = time.time()

# =============================================================================
# 0 - open point cloud file 
pcd = o3d.io.read_point_cloud(pcd_path)
# paint point cloud gray
pcd.paint_uniform_color([0.5, 0.5, 0.5])


# =============================================================================
# 1 - perform Voxel Grid Downsampling
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                                  
# =============================================================================
# 2 Segement ground plane from the rest of the point cloud
_, inliers = pcd.segment_plane(distance_threshold=ransac_dist_thresh,
                               ransac_n=ransac_n,
                               num_iterations=ransac_iters)

# divide segmented Point Cloud into 2 PointCloud objects
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)


# =============================================================================
# 3 - Use DBSCAN to detect objects 
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(outlier_cloud.cluster_dbscan(eps=dbscan_eps, 
                                                   min_points=min_clust, 
                                                   print_progress=False))
# get indiviudal clusters from the labels
cluster_labels, clusters = get_clusters_from_labels(
                                    outlier_cloud, 
                                    labels, 
                                    include_outliers=True)
 
# get unclustered points and set their color to gray
unclustered = clusters[np.where(cluster_labels == -1)[0][0]]
unclustered.paint_uniform_color([0.5, 0.5, 0.5])

# convert to list for display
unclustered = [unclustered]


# =============================================================================
# 4/5 - get 3D bounding boxes

# get bounding box for each cluster
cluster_bboxes = []
for cluster in clusters:
    if len(cluster.points) >= min_clust and len(cluster.points) <= max_clust:
        cluster_bboxes.append(get_axis_aligned_bbox(cluster))

# =============================================================================
# 6 - Combine point cloud objects into single list for visualization


combined_pcd = [inlier_cloud] + clusters + cluster_bboxes

toc = time.time()
print(f'Time to Detect: {toc - tic}')


# visualize
o3d.visualization.draw_geometries(combined_pcd)





