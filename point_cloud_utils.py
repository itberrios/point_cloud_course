"""
Helper functions to get individual clusters
"""
import copy
import numpy as np
from scipy import stats
import open3d as o3d
import matplotlib.pyplot as plt



def get_clusters_from_labels(pcd, labels, include_outliers=True):
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
    # get colors 
    max_label = labels.max()
    colors = plt.get_cmap('tab20')
    colors = colors(labels/(max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    
    unique_labels = np.sort(np.unique(labels))
    
    # remove -1 (unclustered label) if outliers are not permitted
    if not include_outliers:
        unique_labels = unique_labels[unique_labels != -1]

    # store cluster point clouds in a list
    clusters = []
    
    # iterate through each unique label
    for label in unique_labels:
        # get index of points and colors
        cluster_idx = np.where(label == labels)[0]
        
        # get cluster color
        color = colors[cluster_idx, :3]
        
        # get cluster points
        cluster = pcd.select_by_index(cluster_idx)
        
        # paint cluster
        cluster.colors = o3d.utility.Vector3dVector(color)
        
        # append to list
        clusters.append(cluster)
    
    return unique_labels, clusters


def get_knn_clusters(pcd, pcd_tree, max_clusters=500, ep=2):
    ''' Obtains clusters using a KD Tree and K-Nearest Neighbors.
        steps:
            select a point at random
            get it's K-nearest neighbors 
            remove from search space all points within a set dist to the 
                original point 
            repeat
        Inputs:
            pcd - open3d PointCloud object
            pcd_tree - open3d PointCloud KDTreeFlann object
            max_clusters (int) Maximum number of clusters to find
            ep=2 (int) maximum distance from center to edge of cluster
        Outputs:
        '''
    

    # initialize the search space
    points_arr = np.asarray(pcd.points)
    search_space = np.arange(0, len(pcd.points))

    # collect clusters in a list
    clusters = []

    for i in range(max_clusters):

        # get a random point and color
        rand_point = points_arr[np.random.randint(0, len(search_space)), :]
        color = np.random.uniform(size=(3,))
        
        # get 300 nearest neighbors of the point
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(rand_point, 300)
        
        # only keep points within a distance of ep

        idx_prune = np.array(idx)[np.array(dist) <= ep]
         
        # prune search space
        search_space = np.setdiff1d(search_space, idx_prune)
        
        # collect cluster in list and paint it
        cluster = pcd.select_by_index(idx_prune[1:])
        cluster.paint_uniform_color(color)
        
        clusters.append(cluster)
        
        if len(search_space) < 20:
            break
        
    return clusters


def get_oriented_bbox_OLD(pcd):
    ''' Function to obtain oriented bounding box for a given cluster 
        Inputs:
            pcd - open3d PointCloud object
        Outputs:
            bbox_out - open3d LineSet object that contains the bounding box
    '''
    pcd_o3d = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_o3d)
    oriented_bbox = bbox.get_oriented_bounding_box()

    # get bounding box corner points
    # use helper array 
    i_arr = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
        ])

    points = oriented_bbox.center + oriented_bbox.extent*i_arr

    # then get lines (order is based on helper array i_arr)
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], 
             [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # get LineSet for bounding box
    color = stats.mode(np.asarray(pcd.colors))[0].squeeze()

    bbox_out = o3d.geometry.LineSet()
    bbox_out.points = o3d.utility.Vector3dVector(points)
    bbox_out.lines = o3d.utility.Vector2iVector(lines)
    bbox_out.paint_uniform_color(color)
    
    return bbox_out
  
  
def get_axis_aligned_bbox_OLD(pcd):
    ''' Function to obtain axis_aligned bounding box for a given cluster 
        Inputs:
            pcd - open3d PointCloud object
        Outputs:
            bbox_out - open3d LineSet object that contains the bounding box
    '''
    pcd_o3d = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd_o3d)
    ax_aligned_bbox = bbox.get_axis_aligned_bounding_box()

    # get bounding box corner points
    # use helper array
    i_arr = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
        ])

    points = ax_aligned_bbox.max_bound*np.flipud(i_arr) \
             + ax_aligned_bbox.min_bound*i_arr

    # then get lines (order is based on helper array i_arr)
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], 
             [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # get LineSet for bounding box
    color = stats.mode(np.asarray(pcd.colors))[0].squeeze()

    bbox_out = o3d.geometry.LineSet()
    bbox_out.points = o3d.utility.Vector3dVector(points)
    bbox_out.lines = o3d.utility.Vector2iVector(lines)
    bbox_out.paint_uniform_color(color)
    
    return bbox_out

def get_axis_aligned_bbox(pcd):
    ''' Function to obtain axis_aligned bounding box for a given cluster 
        Inputs:
            pcd - open3d PointCloud object
        Outputs:
            bbox_out - open3d AxisAlignedBoundingBox object
    '''
    # get bounding box
    bbox_out = pcd.get_axis_aligned_bounding_box()

    # paint bounding box
    color = stats.mode(np.asarray(pcd.colors))[0].squeeze()
    bbox_out.color = color
    
    return bbox_out