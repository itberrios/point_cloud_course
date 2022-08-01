"""
Script to detect objects in a point cloud
The supported extension names are: pcd, ply, xyz, xyzrgb, xyzn, pts.

This is a faster implementation that 'point_cloud_detect', this script is 
intended to quickly detect objects in the point cloud and return them. It does
not provide any colors of useful visualizations. It is meant for 
"background detection". If a detected object shows some interest it can be 
displayed via other means using the outputs of this script.

This includes a function that is still too slow for realtime implementation.
With appropriate parameter selections 10-20Hz is a realistic speed.

Reading in the point cloud object takes about 0.01 seconds on a PC with 
16GB RAM and a MHz Processor
       
"""
import time
import numpy as np
import open3d as o3d


# =============================================================================
# helper functions
def get_clusters_from_labels(pcd, labels):
    ''' Obtains a list of individual cluster point clouds and paints them 
        unique colors. Assumes that the pcd object has a uniform color.
        Inputs:
            pcd - open3d PointCloud object
            labels - (Nx1 array) labels for each point in the cluster
            include_outliers - (_Bool) determines whether outlaiers should be 
                               included in the output list
        Outputs:
            unique_labels (list) Contains all labels for each cluster
            clusters (list) Contains PointCloud objects for each color
        '''
    unique_labels = np.sort(np.unique(labels))
    
    # remove -1 (unclustered label) to remove outliers
    unique_labels = unique_labels[unique_labels != -1]

    # store cluster point clouds in a list
    clusters = []
    
    # iterate through each unique label
    for label in unique_labels:
        # get cluster points
        clusters.append(pcd.select_by_index(np.where(label == labels)[0]))
    
    
    return clusters


def point_cloud_detect(pcd, voxel_size=0.1, ransac_dist_thresh=0.25, 
                       ransac_iters=500, ransac_n=3, dbscan_eps=0.5, 
                       min_clust=30, max_clust=2000):
    ''' Detects objects in a point cloud file
    
        uses unsupervised learning to segment the point cloud and find
        objects of interest. It does this in 6 main steps
            1) Perform Voxel Downsampling to decrease the total number of 
                   points
            2) Perform planar segmentation via RANSAC to determine the surface 
                   VS all other parts of the point cloud
            3) Perform DBSCAN clustering to find objects of interest
            4) Use PCA to find the axial orienation of each clustered object 
                   (Axial orientation relative to the origin 
                    i.e. 1st principle component)
            5) Obtain bounding boxes for each clustered object with correct 
                   axis alignment obtained in step 4
                   (open3d handles steps 4 and 5 with 
                    'get_axis_aligned_bbox(pcd)')
            6) Return object bounding box info
    
        Inputs:
            pcd open3d PointCloud object
            voxel_size (float) Size of voxel for downsampling
            ransac_dist_thresh (float) distance metric for RANSAC segmentation           
            ransac_iters (int) Number of iterations for RANSAC segmentation
            ransac_n (int) min points to define a plane, best left at 3 or 4
            dbscan_eps (float) DBSCAN cluster neighborhood distance
            min_clust (int) minimum points required to declare a cluster
            max_clust (int) mmaximum points to declare a cluster
        Outputs:
            cluster_bboxes (list) open3d AxisAlignedBoundingBox objects for all
                               detected objects
            
        Future update may include handling of either a point cloud file o
            or open3d PointCloud object
        '''
    # =========================================================================
    # 1 - perform Voxel Grid Downsampling
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                                      
    # =========================================================================
    # 2 Segement ground plane from the rest of the point cloud
    _, inliers = pcd.segment_plane(distance_threshold=ransac_dist_thresh,
                                   ransac_n=ransac_n,
                                   num_iterations=ransac_iters)
    
    # divide segmented Point Cloud into 2 PointCloud objects
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    
    # =========================================================================
    # 3 - Use DBSCAN to detect objects 
    labels = np.array(outlier_cloud.cluster_dbscan(eps=dbscan_eps, 
                                                   min_points=min_clust, 
                                                   print_progress=False))
    # get indiviudal clusters from the labels
    clusters = get_clusters_from_labels(outlier_cloud,
                                        labels)
     
    
    
    # =========================================================================
    # 4/5 - get 3D bounding boxes
    
    # get bounding box for each cluster
    cluster_bboxes = []
    for cluster in clusters:
        if len(cluster.points) <= max_clust:
            cluster_bboxes.append(cluster.get_axis_aligned_bounding_box())
    
    return cluster_bboxes


# =============================================================================
# test program excecution

def main():
    pcd_path = r'test_files\sdc.pcd'
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    tic = time.time()
    cluster_bboxes = point_cloud_detect(pcd, voxel_size=0.15, 
                       ransac_dist_thresh=0.25, ransac_iters=150, ransac_n=3, 
                       dbscan_eps=0.5, min_clust=30, max_clust=2000)

    toc = time.time()
    print(f'Time to Detect: {toc - tic}')
    
    print(f'Number of objects detected: {len(cluster_bboxes)}')

if __name__ == '__main__':
    main()
    
    
    
    
    
