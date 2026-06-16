import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize,skeletonize_3d

import os
import time


"""
这里有这么几层
points-array[3,N],存储的是点云坐标
graph-list[GraphNode],存储的是节点,与点云坐标的数量一致,但是顺序不一致
node-GraphNode,成员变量current_index是当前节点对应的点在points中的索引
               成员变量adj_points存放的是当前节点的近邻节点在graph中的索引
"""

class GraphNode:
    def __init__(self,point_index) -> None:
        self.current_point_index = point_index

        #adj_points里存储的是近邻点在graph列表中的排序
        self.adj_node_indices = []
        self.num_adj_points = 0

        #是否被搜索过的标签
        self.flag = 0
    
    def add_adjacent_point(self,node_index):
        self.adj_node_indices.append(node_index)
        self.num_adj_points = self.num_adj_points+1

    def remove_adjacent_point(self,node_index):
        
        self.adj_node_indices.remove(node_index)
        self.num_adj_points = self.num_adj_points-1

def graph_to_adj_matrix(graph:list[GraphNode]):

    """邻接矩阵,adj_matrix[i,j]代表第i个点的近邻节点是否有第j个点"""
    adj_matrix = np.ones([len(graph),len(graph)])
    for node in graph:
        start_index = node.current_point_index
        for adj_node_index in node.adj_node_indices:
            adj_node = graph[adj_node_index]
            end_index = adj_node.current_point_index
            adj_matrix[start_index, end_index] = 1
            adj_matrix[end_index, start_index] = 1

    return adj_matrix



def create_graph(origin_points:np.ndarray,begin_point_index)->list[GraphNode]:

    """
    Input: 
    origin_points-[3,N]
    Return:
    graph-这是一个列表,由多个节点组成,每个节点包含当前点在origin_points中的序号和其近邻点的序号
    """

    _,num_points = origin_points.shape

    #ndarray-[N]
    unorder_indices = np.arange(0,num_points)

    #ndarray-[3,N]
    unorder_points = origin_points.copy()
    graph_points = []
    graph = []

    """以最下面的点作为初始起点"""
    #一个数
    #begin_point_index = np.argmin(origin_points[-1])
    #[3,1]
    begin_point = np.expand_dims(origin_points[:,begin_point_index],axis=1)
    begin_node = GraphNode(begin_point_index)
    graph.append(begin_node)
    graph_points.append(origin_points[:,begin_point_index])

    
    unorder_indices = np.delete(unorder_indices, begin_point_index)
    unorder_points = np.delete(unorder_points, begin_point_index, axis=1)

    while 1:
        """计算当前unoreder_points到begin_node的最小距离"""
        #dist_to_begin-[N']
        dist_to_begin = np.sum((unorder_points-begin_point)**2,axis=0)
        index = np.argmin(dist_to_begin)
        adj_dist = dist_to_begin[index]

        """如果距离大于近邻范围,则需要重新选择begin_node和并计算它的近邻点"""
        if adj_dist > 4:
            """计算unoreder_points到graph_points之间的最小距离点对"""
            graph_points_array = np.array(graph_points).T
            _,K = unorder_points.shape
            mini = 10000
            for k in range(K):
                #计算unorder_points中每个点到graph_points的距离
                p = np.expand_dims(unorder_points[:,k],axis=1)
                dist = np.sum((graph_points_array-p)**2,axis=0)
                dist_min = np.min(dist)
                if dist_min < mini:
                    #更新begin_node和近邻点在unorder_points中的index
                    index = k
                    begin_node_index = np.argmin(dist)
                    mini = dist_min
            
            begin_node = graph[begin_node_index]


        """将近邻点放入拓扑点集中"""
        adj_point_index = unorder_indices[index]
        adj_point = unorder_points[:,index]
        #加入到点集
        graph_points.append(adj_point)
        #加入到图中
        adj_node = GraphNode(adj_point_index)
        graph.append(adj_node)
        #将其链接到begin_node的近邻点上
        begin_node.add_adjacent_point(len(graph)-1)

        """并将其从未排序点集中取出"""
        unorder_indices = np.delete(unorder_indices, index)
        unorder_points = np.delete(unorder_points, index, axis=1)


        """判断原始点集是否为空"""
        if unorder_points.shape[1] == 0:
            break 

        """以拓扑点集的最后一个点为新的起点"""
        begin_point = np.expand_dims(graph_points[-1],axis=1)
        begin_node = graph[-1]
    
    return graph


def if_node_depth_larger_than_k(graph:list[GraphNode],node_index:int, k:int):

    node = graph[node_index]
    if k == 0:
        return True
    elif node.num_adj_points == 0:
        return False
    else:
        for adj_node_index in node.adj_node_indices:
            branch_flag = if_node_depth_larger_than_k(graph, adj_node_index, k-1)
            if branch_flag:
                break

        return branch_flag
    
def remove_noise_points(skele_graph:list[GraphNode],k:int):

    """
    Input:
    skele_graph-血管中心线的拓扑图
    k-当分叉点的子节点深度低于k时,则断开子节点的连接
    """
    for node in skele_graph:
        #将近邻节点数超过2的节点取出
        if node.num_adj_points >= 2:
            for adj_node_index in node.adj_node_indices:
                #计算分叉点的所有子节点的深度
                if not if_node_depth_larger_than_k(skele_graph, adj_node_index, k):
                    #如果该子节点深度不够,则断开当前节点node与该子节点的连接,
                    #这样后续从根节点遍历图时就不会经过它
                    node.remove_adjacent_point(adj_node_index)

def obtain_branches(graph:list[GraphNode], begin_node_index, branch=[], branches=[]):

    """
    采用递归的方式进行深度优先搜索
    Input: 
    graph-待分支的图
    begin_node_index-分支的根节点在graph中的索引
    branch-[node_index]用于存放分支中包含的节点索引
    branches-[branch]用于存放branch, 需用户给定
    """

    node_index = begin_node_index
    while 1:
        node = graph[node_index]
        if node.num_adj_points >= 2:
            branches.append(branch)
            for adj_node_index in node.adj_node_indices:
                obtain_branches(graph,adj_node_index,[],branches)
            
            return
        elif node.num_adj_points == 0:
            branches.append(branch)
            
            return
        elif node.num_adj_points == 1: 
            branch.append(node_index)
            node_index = node.adj_node_indices[0]
        
def get_point_indices_from_branches(graph:list[GraphNode],branches:list[list]):
    point_indices = []

    for branch in branches:
        point_indices_in_branch = []
        for node_index in branch:
            node = graph[node_index]
            point_indices_in_branch.append(node.current_point_index)
        point_indices.append(point_indices_in_branch)
    return point_indices




def get_topo_weights(skele_points):

    """
    Input: 
    skele_points-[3,N], 血管中心线点坐标
    Output: 
    topology_weight-[N], 与skele_points中对应的各个血管中心线的拓扑权重
    """

    """构建拓扑图, 每次连接未连接点集中最近的一个点"""
    skele_graph = create_graph(skele_points,0)

    """确定分叉点"""
    #用于存储分叉点在skele_graph中的索引
    bifur_node_indices = []
    #将近邻节点数超过2的节点取出，计算每个子节点的深度，如果超过阈值k,则将其认定为一个分叉点
    for k,node in enumerate(skele_graph):
        if node.num_adj_points >= 2:
            branch = 0
            for adj_node_index in node.adj_node_indices:
                adj_node = skele_graph[adj_node_index]
                if if_node_depth_larger_than_k(skele_graph, adj_node_index, 5):
                    branch = branch+1
            if branch >= 2:
                bifur_node_indices.append(k)

    """赋权"""
    bifur_points_indices = []
    for bifur_index in bifur_node_indices:
        bifur_node = skele_graph[bifur_index]
        bifur_points_indices.append(bifur_node.current_point_index)
    topology_weight = np.ones(skele_points.shape[1])
    topology_weight[bifur_points_indices] = 2

    return skele_graph, bifur_node_indices, topology_weight




if __name__ == "__main__":

    """读取三维影像"""
    #读取影像数据
    dsa3d_seg_path = "F:\\3D_2D\\477583XA20201113\\3D_DSA_seg.nii.gz"
    seg_ia_img_dsa3d = sitk.ReadImage(dsa3d_seg_path)
    seg_ia_array_dsa3d = sitk.GetArrayFromImage(seg_ia_img_dsa3d).transpose(2,1,0)
    space_3d = seg_ia_img_dsa3d.GetSpacing()
    size_3d = seg_ia_img_dsa3d.GetSize()

    #生成T_3d_voxel_to_img
    T_3d_voxel_to_image_dsa3d = np.array([[space_3d[0],0          ,0          ,-(size_3d[0]-1)*space_3d[0]/2],
                                          [0          ,space_3d[1],0          ,-(size_3d[1]-1)*space_3d[1]/2],
                                          [0          ,0          ,space_3d[2],-(size_3d[2]-1)*space_3d[2]/2],
                                          [0          ,0          ,0          ,1                            ]])


    """获取血管中心线的三维坐标"""
    #细化三维血管
    skeleton_3d = skeletonize_3d(seg_ia_array_dsa3d)
    #获取中心线三维坐标
    skeleton_3d_xyz = np.argwhere(skeleton_3d>0)#[N,3]
    s1_0 = np.column_stack((skeleton_3d_xyz,np.ones((len(skeleton_3d_xyz),1)))).T #[4,N]
    #将中心线三维坐标
    s1_0 = T_3d_voxel_to_image_dsa3d@s1_0

    graph, bifur_node_indices, topology_weights = get_topo_weights(s1_0[0:3])

    #根据bifur_indices取出对应的点的坐标
    bifur_points = []
    for bifur_node_index in bifur_node_indices:
        bifur_node = graph[bifur_node_index]
        bifur_points.append(s1_0[0:3,bifur_node.current_point_index])
    bifur_points = np.array(bifur_points).T

    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")

    ax_min = np.floor(s1_0[0:3].min(axis = 1)/10)*10
    ax_max = np.ceil(s1_0[0:3].max(axis = 1)/10)*10
        
    #显示的时候需要调转一下方向
    ax.set_xticks(np.linspace(ax_min[0], ax_max[0], int((ax_max[0]-ax_min[0])/5))+1)
    ax.set_yticks(np.linspace(ax_min[1], ax_max[1], int((ax_max[1]-ax_min[1])/5))+1)
    ax.set_zticks(np.linspace(ax_min[2], ax_max[2], int((ax_max[2]-ax_min[0])/5))+1)
    ax.set_box_aspect([ax_max[0] - ax_min[0],ax_max[1] - ax_min[1],ax_max[2] - ax_min[2]])

    ax.scatter(s1_0[0],s1_0[1],s1_0[2])
    ax.scatter(bifur_points[0],bifur_points[1],bifur_points[2],)

    for node in graph:
        start_point = s1_0[0:3,node.current_point_index]
        for adj_index in node.adj_node_indices:
            end_point = s1_0[0:3,graph[adj_index].current_point_index]
            ax.quiver(start_point[0],
                      start_point[1],
                      start_point[2],
                      end_point[0]-start_point[0],
                      end_point[1]-start_point[1],
                      end_point[2]-start_point[2])


    plt.show()

