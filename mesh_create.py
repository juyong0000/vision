import open3d as o3d
import numpy as np
import copy

#parameter
width = 3
depth= 0.155
height = 4
radius = 0.032
truncated_cone_height = 4

move_x = 0.03
move_y = 0.01
move_z = 0.41

rot_x = np.pi/3
rot_y = 0
rot_z = np.pi/4

#shape
mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=depth, resolution=100, split=4, create_uv_map=False)
# mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth, create_uv_map=False, map_texture_to_each_face=False)
# mesh_cropped =o3d.geometry.OrientedBoundingBox
mesh_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.038, height=0.44, resolution=20, split=2, create_uv_map=False)
# mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20, create_uv_map=False)
coordinate=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=([0.0, 0.0, 0.0]))
#rotation
# R = mesh.get_rotation_matrix_from_xyz((rot_x, rot_y, rot_z))


R1 = [[0.39, -0.89, 0.24],
    [0.81, 0.45, 0.37],
    [-0.43, 0.05, 0.90]]

#cone
R2 = [[-0.8314, 0.5549, 0.4354],
    [-0.5336, -0.8178, 0.2156],
    [0.155, 0.156, 0.9755 ]]

# print(mesh1.triangles)
# o3d.visualization.draw_geometries([mesh1],mesh_show_wireframe=True)
T_first_cone = np.eye(4)
T_first_cone[:4,:4] =[[0,0,-1,0.4406],
                      [0,1,0,0],
                      [1,0,0,0],
                      [0,0,0,1]]

mesh_f_cone = copy.deepcopy(mesh_cone).transform(T_first_cone)


T_cone = [
    [-0.83, 0.55, 0.44,0.24],
    [-0.53, -0.82, 0.22,0.23],
    [0.16, 0.16, 0.98, 0.34 ],
    [0,    0,    0,    1]]

print(np.dot(T_cone,[0.44,0,0,1]))

#transform for cylinder
T_first = np.eye(4)
T_first[:4,:4] =[[0,0,1,0],
                 [0,1,0,0],
                 [-1,0,0,0],
                 [0,0,0,1]]
mesh_f = copy.deepcopy(mesh).transform(T_first)

T = np.eye(4)
# T[:3, :3] = mesh.get_rotation_matrix_from_xyz((rot_x, rot_y, rot_z))

T[:3,:3] = [[0.39, -0.89, 0.24],
            [0.81, 0.45, 0.37],
            [-0.43, 0.05, 0.90]]
T[0, 3] = move_x
T[1, 3] = move_y
T[2, 3] = move_z


T= T.round(3)

axis_z = [0,0,1]
axis_y = [0,1,0]
axis_x = [1,0,0]


new_axis_z = np.dot(R1,axis_z).round(3)
new_axis_y = np.dot(R1,axis_y).round(3)
new_axis_x = np.dot(R1,axis_x).round(3)

new_axis_z_cone = np.dot(R2,axis_z).round(3)
new_axis_y_cone = np.dot(R2,axis_y).round(3)
new_axis_x_cone = np.dot(R2,axis_x).round(3)

mesh_t = copy.deepcopy(mesh_f).transform(T)
mesh_t.compute_vertex_normals()
pcd = mesh_t.sample_points_poisson_disk(5000)


mesh_t_cone = copy.deepcopy(mesh_f_cone).transform(T_cone)

mesh_f_cone.compute_vertex_normals()
pcd_cone = mesh_f_cone.sample_points_poisson_disk(5000)

# pcd = mesh.sample_points_poisson_disk(5000)



# 500번째 점에서 거리 1 미만인 점 찾기
# Flann_pcd = o3d.geometry.KDTreeFlann(pcd)
# [k, idx, _] = Flann_pcd.search_radius_vector_3d(pcd.points[500], 1)
# np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
# o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True, point_show_normal=True)

# 500번째 점에서 거리 가장 가까운 4개 점

x=0
y=0
z=0

x_cone=0
y_cone=0
z_cone=0

points = np.asarray(pcd_cone.points)
mask = points[:,0] > 0.328
pcd_cone.points = o3d.utility.Vector3dVector(points[mask])

for i in range(0,len(np.asarray(pcd_cone.points))):
    
    pcd_cone.points[i] = np.dot(R2,pcd_cone.points[i])+[0.243,0.229,0.341]
    
pcd.paint_uniform_color([1,0,0])
Flann_pcd = o3d.geometry.KDTreeFlann(pcd)
pcd.estimate_normals()

pcd_cone.paint_uniform_color([1,0,0])
Flann_pcd_cone = o3d.geometry.KDTreeFlann(pcd_cone)
pcd_cone.estimate_normals()

for i in range(0,len(np.asarray(pcd_cone.points))):

    [k, idx, _] = Flann_pcd_cone.search_knn_vector_3d(pcd_cone.points[i], 5)
 
    #코드수정
    point_1=np.asarray(pcd_cone.normals[idx[1]])
    point_2=np.asarray(pcd_cone.normals[idx[2]])
    point_3=np.asarray(pcd_cone.normals[idx[3]])
    point_4=np.asarray(pcd_cone.normals[idx[4]])

    #4개 점 2개씩 내적
    result1=abs(np.inner(point_1,point_2))
    result2=abs(np.inner(point_1,point_3))
    result3=abs(np.inner(point_1,point_4))
    result4=abs(np.inner(point_2,point_3))
    result5=abs(np.inner(point_2,point_4))
    result6=abs(np.inner(point_3,point_4))

    final_result= (result1+result2+result3+result4+result5+result6)/6

    x_cone+=pcd_cone.points[i][0]
    y_cone+=pcd_cone.points[i][1]
    z_cone+=pcd_cone.points[i][2]

    if (final_result > 0.95):
        np.asarray(pcd_cone.colors)[i] = [0, 1, 0]

for i in range(0,len(np.asarray(pcd.points))):

    [k, idx, _] = Flann_pcd.search_knn_vector_3d(pcd.points[i], 5)
    # [k, idx, _] = Flann_pcd.search_radius_vector_3d(pcd.points[i], 1)
    
    #코드수정
    # np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]
    point_1=np.asarray(pcd.normals[idx[1]])
    point_2=np.asarray(pcd.normals[idx[2]])
    point_3=np.asarray(pcd.normals[idx[3]])
    point_4=np.asarray(pcd.normals[idx[4]])

    #4개 점 2개씩 내적
    result1=abs(np.inner(point_1,point_2))
    result2=abs(np.inner(point_1,point_3))
    result3=abs(np.inner(point_1,point_4))
    result4=abs(np.inner(point_2,point_3))
    result5=abs(np.inner(point_2,point_4))
    result6=abs(np.inner(point_3,point_4))

    final_result= (result1+result2+result3+result4+result5+result6)/6

    x+=pcd.points[i][0]
    y+=pcd.points[i][1]
    z+=pcd.points[i][2]

    if (final_result > 0.95):
        np.asarray(pcd.colors)[i] = [0, 1, 0]

centroid_x=x/len(np.asarray(pcd.points))
centroid_y=y/len(np.asarray(pcd.points))
centroid_z=z/len(np.asarray(pcd.points))

centroid_x_cone=x_cone/len(np.asarray(pcd_cone.points))
centroid_y_cone=y_cone/len(np.asarray(pcd_cone.points))
centroid_z_cone=z_cone/len(np.asarray(pcd_cone.points))



# vector_from_c_x = [centroid_x+(depth/2)*new_axis_x[0],centroid_y+(depth/2)*new_axis_x[1],centroid_z+(depth/2)*new_axis_x[2]]
# vector_from_c_y = [centroid_x+(depth/2)*new_axis_y[0],centroid_y+(depth/2)*new_axis_y[1],centroid_z+(depth/2)*new_axis_y[2]]
# vector_from_c_z = [centroid_x+(depth/2)*new_axis_z[0],centroid_y+(depth/2)*new_axis_z[1],centroid_z+(depth/2)*new_axis_z[2]]

for i in range(0,len(np.asarray(pcd.points))):
    #shape에 따라 point 어떻게 잡을지...
    [kz1, idx_z1, _] = Flann_pcd.search_radius_vector_3d([centroid_x+(depth/2)*new_axis_x[0],centroid_y+(depth/2)*new_axis_x[1],centroid_z+(depth/2)*new_axis_x[2]],radius/2)
    [kz2, idx_z2, _] = Flann_pcd.search_radius_vector_3d([centroid_x-(depth/2)*new_axis_x[0],centroid_y-(depth/2)*new_axis_x[1],centroid_z-(depth/2)*new_axis_x[2]],radius/2)

    [k_radius, idx_radius, _ ]= Flann_pcd.search_radius_vector_3d([centroid_x,centroid_y,centroid_z],radius*1.02)

    # [kx1,idx_x1, _] = Flann_pcd.search_radius_vector_3d([centroid_x+(width/2)*new_axis_x[0],centroid_y+(width/2)*new_axis_x[1],centroid_z+(width/2)*new_axis_x[2]],radius/2)
    # [kx2,idx_x2, _] = Flann_pcd.search_radius_vector_3d([centroid_x-(width/2)*new_axis_x[0],centroid_y-(width/2)*new_axis_x[1],centroid_z-(width/2)*new_axis_x[2]],radius/2)

    # [ky1,idx_y1, _] = Flann_pcd.search_radius_vector_3d([centroid_x+(height/2)*new_axis_y[0],centroid_y+(height/2)*new_axis_y[1],centroid_z+(height/2)*new_axis_y[2]],radius/2)
    # [ky2,idx_y2, _] = Flann_pcd.search_radius_vector_3d([centroid_x-(height/2)*new_axis_y[0],centroid_y-(height/2)*new_axis_y[1],centroid_z-(height/2)*new_axis_y[2]],radius/2)

    # [k_cone, idx_cone, _] = Flann_pcd.search_radius_vector_3d([move_x,move_y,move_z],radius/2)
    # [k_cone_z, idx_cone_z, _] = Flann_pcd.search_radius_vector_3d([move_x+depth*new_axis_z[0]/4,move_y+depth*new_axis_z[1]/4,move_z+depth*new_axis_z[2]/4],3*radius/4)
    
    # [k_hollow_cone, idx_hollow_cone, _] = Flann_pcd.search_radius_vector_3d([move_x,move_y,move_z],radius/2)
    # [k_hollow_cone_z, idx_hollow_cone_z, _] = Flann_pcd.search_radius_vector_3d([move_x+depth*new_axis_z[0]/3,move_y+depth*new_axis_z[1]/3,move_z+depth*new_axis_z[2]/3],2*radius/3)
    

    if np.all(pcd.colors[i] == [0, 1, 0]) :
        
        # for cylinder
        np.asarray(pcd.colors)[idx_z1] = [0, 0, 1]
        np.asarray(pcd.colors)[idx_z2] = [0, 0, 1]
        np.asarray(pcd.colors)[idx_radius] = [0, 0, 1]

        # # for box
        # np.asarray(pcd.colors)[idx_x1] = [0, 0, 1]
        # np.asarray(pcd.colors)[idx_x2] = [0, 0, 1]
        # np.asarray(pcd.colors)[idx_y1] = [0, 0, 1]
        # np.asarray(pcd.colors)[idx_y2] = [0, 0, 1]

        #for cone
        # np.asarray(pcd.colors)[idx_cone] = [0, 0, 1]
        # np.asarray(pcd.colors)[idx_cone_z] = [0, 0, 1]

        # for hollow cone
        # np.asarray(pcd.colors)[idx_hollow_cone] = [0, 0, 1]
        # np.asarray(pcd.colors)[idx_hollow_cone_z] = [0, 0, 1]
for i in range(0,len(np.asarray(pcd_cone.points))):
   
    [k_cone, idx_cone, _] = Flann_pcd_cone.search_radius_vector_3d([-0.125,-0.003,0.4104],0.038/2)
    [k_cone_z, idx_cone_z, _] = Flann_pcd_cone.search_radius_vector_3d([-0.076, 0.03, 0.4],0.037)

    if np.all(pcd_cone.colors[i] == [0, 1, 0]) :
        #for cone
        np.asarray(pcd_cone.colors)[idx_cone] = [0, 0, 1]
        np.asarray(pcd_cone.colors)[idx_cone_z] = [0, 0, 1]



o3d.visualization.draw_geometries([pcd,pcd_cone,coordinate], mesh_show_wireframe=True, point_show_normal=True)


