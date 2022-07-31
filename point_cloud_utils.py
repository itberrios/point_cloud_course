"""
Helper functions to get individual clusters
"""
import numpy as np
import time
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt



def get_clusters_from_labels(pcd, labels, background_color=[0,0,0]):
    ''' Obtains a list of individual cluster point clouds and paints them 
        unique colors. Assumes that the pcd object has a uniform color.
        Inputs:
            pcd - open3d PointCloud object
            labels - (Nx1 array) labels for each point in the cluster
            background_color - (lust/array) original background color of pcd
        Outputs:
            clusters (list) Contains PointCloud objects for each color
        '''
    # sanitize inputs
    if not isinstance(background_color, np.ndarray):
        background_color = np.array(background_color)
        
    # get colors 
    max_label = labels.max()
    colors = plt.get_cmap('tab20')
    colors = colors(labels/(max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    
    # get unique labels and remove -1 (unclustered label)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    # store cluster point clouds in a list
    clusters = []
    
    # iterate through each unique label
    for label in unique_labels:
        # get index of points and colors
        cluster_idx = np.where(label == labels)[0]
        
        # get cluster color
        color = colors[cluster_idx, :3]
        
        # don't add unclustered points
        # if color.sum() == background_color.sum():
        #     continue
        
        # get cluster points
        cluster = pcd.select_by_index(cluster_idx)
        
        # paint cluster
        cluster.colors = o3d.utility.Vector3dVector(color)
        
        # append to list
        clusters.append(cluster)
    
    return clusters


def get_knn_clusters(pcd, pcd_tree, max_points=500, ep=2):
    ''' Obtains clusters using a KD Tree and K-Nearest Neighbors.
        steps:
            select a point at random
            get it's K-nearest neighbors 
            remove from search space all points within a set dist to the 
                original point 
            repeat
        Inputs:
        Outputs:
        '''
    

    # initialize the search space
    points_arr = np.asarray(pcd.points)
    search_space = np.arange(0, len(pcd.points))

    # collect clusters in a list
    clusters = []

    for i in range(max_points):

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
    