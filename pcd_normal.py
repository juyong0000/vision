import open3d as o3d
import numpy as np


pcd = o3d.io.read_point_cloud("apple.ply")

pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True, point_show_normal=True)