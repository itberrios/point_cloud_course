"""
Helper functions to get individual clusters
"""
import numpy as np
import time
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt


def get_clusters(pcd, colors, background_color=[0,0,0]):
    ''' Obtains a list of individual cluster point clouds and paints them 
        unique colors. 
        This function doesn;t actually get the clusters... it just gets the 
        clusters based on their color. The new function will use labels to 
        get the individual clusters in a list
        Inputs:
            pcd - open3d PointCloud object
            colors - (Nx3 array) Contains colors for each cluster
            background_color - (lust/array) original background color of pcd
        Outputs:
            cluster_clouds (list) Contains PointCloud objecrts for each color
        '''
    # sanitize inputs
    if not isinstance(background_color, np.ndarray):
        background_color = np.array(background_color)
        
    # each unique color value corresponds to a different cluster/category
    clusters = colors[:, :3].sum(axis=1)

    sorted_clusters = np.sort(clusters)
    unique_colors = np.unique(colors[:, :3], axis=0)
    
    # this allows us to get the correct cluster RGB color
    cluster_indexes = np.argsort(unique_colors.sum(axis=1))


    # get cluster locations as a 2D list (or 2D array?)
    cluster_clouds = []
    for i, clust in enumerate(np.unique(sorted_clusters)):
        
        # get cluster color and point cloud locations
        cluster_color = unique_colors[cluster_indexes[i], :3]
        
        # don't add unclustered points
        if cluster_color.sum() == background_color.sum():
            continue
        
        # get cluster point cloud 
        cluster_loc = np.where(clusters == clust)[0].tolist()
        
        # get current cluster point cloud with color
        cluster_cloud = pcd.select_by_index(cluster_loc)
        cluster_cloud.paint_uniform_color(cluster_color)
        
        # append to list
        cluster_clouds.append(cluster_cloud)
        

    return cluster_clouds
    
# new way
def get_clusters_2(pcd, labels, background_color=[0,0,0]):
    ''' Obtains a list of individual cluster point clouds and paints them 
        unique colors
        Inputs:
            pcd - open3d PointCloud object
            colors - (Nx1 array) labels for each point in the cluster
            background_color - (lust/array) original background color of pcd
        Outputs:
            cluster_clouds (list) Contains PointCloud objecrts for each color
        '''

    # get colors 
    unique_labels = np.unique(labels)
    max_label = labels.max()
    colors = plt.get_cmap('tab20')(unique_labels / (max_label if max_label > 0 else 1))
    colors[unique_labels < 0] = 0
    
    
    # each unique color value corresponds to a different cluster/category
    clusters = colors[:, :3].sum(axis=1)

    sorted_clusters = np.sort(clusters)
    unique_colors = np.unique(colors[:, :3], axis=0)
    
    # this allows us to get the correct cluster RGB color
    cluster_indexes = np.argsort(unique_colors.sum(axis=1))


    # get cluster locations as a 2D list (or 2D array?)
    cluster_clouds = []
    for i, clust in enumerate(np.unique(sorted_clusters)):
        
        # get cluster color and point cloud locations
        cluster_color = unique_colors[cluster_indexes[i], :3]
        
        # don't add unclustered points
        if cluster_color.sum() == background_color.sum():
            continue
        
        # get cluster point cloud 
        cluster_loc = np.where(clusters == clust)[0].tolist()
        
        # get current cluster point cloud with color
        cluster_cloud = pcd.select_by_index(cluster_loc)
        cluster_cloud.paint_uniform_color(cluster_color)
        
        # append to list
        cluster_clouds.append(cluster_cloud)
        

    return cluster_clouds


    
'''
# Example for how to get clusters and colors


with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.5, 
                                                   min_points=12, 
                                                   print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

# get cluster colors
colors = plt.get_cmap('tab20')(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0

'''