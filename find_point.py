import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d
import copy

class Find_point:
    def __init__(self):
        self.height
        self.radius
        self.shape
        self.points
        self.truncated_cone_height
        self.axis_z = [0,0,1]
        self.axis_y = [0,1,0]
        self.axis_x = [1,0,0]
    def create_mesh(self,radius,height,shape):
        if shape =='cylinder':
            self.mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=100, split=4, create_uv_map=False)
        elif shape == 'cone': 
            self.mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=20, split=2, create_uv_map=False)
        elif shape == 'sphere': 
            self.mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20, create_uv_map=False)
        return self.mesh

    def create_pcd(self,trans_m,shape,mesh):

        if shape == 'cylinder': 
            t_init = np.eye(4)
            t_init[:4,:4] =[[0,0,1,0],
                            [0,1,0,0],
                            [-1,0,0,0],
                            [0,0,0,1]]
            mesh_init = copy.deepcopy(mesh).transform(t_init)
            mesh_final = copy.deepcopy(mesh_init).transform(trans_m)
            mesh_final.compute_vertex_normals()
            self.pcd = mesh_final.sample_points_poisson_disk(5000)

        if shape == 'cone':    
            t_init = np.eye(4)
            t_init[:4,:4] =[[0,0,-1,self.radius],
                            [0,1,0,0],
                            [1,0,0,0],
                            [0,0,0,1]]
            mesh_init = copy.deepcopy(mesh).transform(t_init)
            mesh_init.compute_vertex_normals()
            self.pcd = mesh_init.sample_points_poisson_disk(5000)

            points = np.asarray(self.pcd.points)
            mask = points[:,0] > self.truncated_cone_height
            self.pcd.points = o3d.utility.Vector3dVector(points[mask])
            
            for i in range(0,len(np.asarray(self.pcd.points))):
    
                self.pcd.points[i] = np.dot(trans_m[:3,:3],self.pcd.points[i])+trans_m[:,3]

        self.new_axis_z = np.dot(trans_m[:3,:3],self.axis_z).round(3)
        self.new_axis_y = np.dot(trans_m[:3,:3],self.axis_y).round(3)
        self.new_axis_x = np.dot(trans_m[:3,:3],self.axis_x).round(3)
     
    def find_plane(self,pcd):
        pcd.paint_uniform_color([1,0,0])
        Flann_pcd = o3d.geometry.KDTreeFlann(pcd)
        pcd.estimate_normals()

        for i in range(0,len(np.asarray(pcd.points))):

            [k, idx, _] = Flann_pcd.search_knn_vector_3d(pcd.points[i], 5)
            
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




if __name__ == '__main__':
    findP = Find_point()

