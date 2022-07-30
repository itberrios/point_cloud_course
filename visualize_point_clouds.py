"""
Visualize Point clouds 

Requires Python 3.7, open3d, and pptk (only supported up to Python 3.7)

Usages: 
    python visualize_point_clouds.py --pcd_path path
    python visualize_point_clouds.py --pcd_path path --viz open3d
    python visualize_point_clouds.py --pcd_path path --viz pptk
"""
import os
import sys
import argparse
import open3d as o3d
import pptk


# TEMP
# pcd_path = r'C:\Users\itber\Documents\learning\think_autonomous_main\point_cloud_course\test_files\n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605047769.pcd'


def main(pcd_path, viz):
    print('Loading Point Cloud... \n')
    if os.path.exists(pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
    else:
        print('Path does not exist! \n')
        print('Exiting... \n')
        sys.exit()
        
    print(pcd)
    # print(np.asarray(pcd.points))

    if viz == 'pptk':
        pptk.viewer(pcd.points)
    else:
        # use open3d as default
        o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcd_path', type=str, help='Path to .pcd file')
    ap.add_argument('--viz', type=str, default='open3d', 
                    help='Visualization API -- open3d or pptk')
    args = vars(ap.parse_args())
    
    main(args['pcd_path'], args['viz'])
    



