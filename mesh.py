import open3d as o3d
import numpy as np

print("Testing mesh in Open3D...")

mesh = o3d.io.read_triangle_mesh("tuna_can.ply")

print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True,point_show_normal=True)

pcd = mesh.sample_points_uniformly(number_of_points=5000)
o3d.visualization.draw_geometries([pcd])
#####################################

pcd = mesh.sample_points_poisson_disk(2000)
# pcd.normals = o3d.utility.Vector3dVector(np.zeros(
#     (1, 3)))  # invalidate existing normals
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True, point_show_normal=True)