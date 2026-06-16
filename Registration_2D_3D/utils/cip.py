import numpy as np

"""累计密度投影,用于投影三维分割图"""
def CIP(vol,T1,T2,P,img_shape):
    """
    Input:
    vol-[H,W,D],输入的三维影像
    T1-[4,4],三维影像像素坐标系到图像坐标系的变换
    T2-[4,4],位姿变换矩阵
    P-[3,4],投影矩阵
    Output:
    img_proj-累计密度投影图
    """
    
    """获取分割点的三维坐标,points_xyz-[4,N]"""
    points_index = np.argwhere(vol)#[N,3]
    points_index = np.concatenate([points_index.T,np.ones((1,points_index.shape[0]))],axis=0)#[4,N]
    points_xyz = T2@T1@points_index#[4,N]

    """投影"""
    points_Proj = P@points_xyz#[3,N]
    points_Proj = points_Proj/np.expand_dims(points_Proj[2],axis=0)#[3,N]
    points_Proj = points_Proj.T#[N,3]

    """获得累积投影图"""
    points_Proj = np.around(points_Proj)
    img_proj = np.zeros(img_shape)
    for point in points_Proj:
        index_x = int(point[0])
        index_y = int(point[1])
        if index_x < img_shape[0] and index_x >= 0 and index_y < img_shape[1] and index_y >= 0:
            img_proj[index_x,index_y] = img_proj[index_x,index_y]+1

    img_proj = np.where(img_proj>0,1,0)
    
    """对累计投影图进行修饰"""
    img_points = np.argwhere(img_proj>0)
    max = np.max(img_points,axis=0)
    min = np.min(img_points,axis=0)

    for i in range(int(min[0]),int(max[0])):
        for j in range(int(min[1]),int(max[1])):
            if img_proj[i,j] == 0:
                if img_proj[i-1,j]==1 and img_proj[i+1,j]==1:
                    img_proj[i,j]=1
                elif img_proj[i,j-1]==1 and img_proj[i,j+1]==1:
                    img_proj[i,j]=1
    return img_proj