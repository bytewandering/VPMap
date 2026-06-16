import numpy as np
import SimpleITK as sitk

def w_to_T(w:np.ndarray):
    #给T补充上最后一行齐次项[0,0,0,1]
    R_x = np.array([[1            ,0            ,0            ],
                    [0            ,np.cos(w[0]) ,-np.sin(w[0])],
                    [0            ,np.sin(w[0]) ,np.cos(w[0]) ]])
    
    R_y = np.array([[np.cos(w[1]) ,0            ,np.sin(w[1]) ],
                    [0            ,1            ,0            ],
                    [-np.sin(w[1]) ,0           ,np.cos(w[1]) ]])
    
    R_z = np.array([[np.cos(w[2]) ,-np.sin(w[2]),0            ],
                    [np.sin(w[2]) ,np.cos(w[2]) ,0            ],
                    [0            ,0            ,1            ]])

    R = R_z@R_y@R_x

    T = np.concatenate([R,np.expand_dims(w[3:6],axis=1)],axis=1)
    T = np.concatenate([T,np.array([[0,0,0,1]])],axis=0)

    return T

"""获取相机的内外参"""
def camera_matrix(img:sitk.Image):
    #读取二维影像的焦距
    f = float(img.GetMetaData("0018|1110"))

    #读取二维影像在x,y方向上的spacing
    space_x = img.GetSpacing()[0]
    space_y = img.GetSpacing()[1]

    #图像中心点在像素坐标系中的坐标
    u = (img.GetSize()[0]-1)/2
    v = (img.GetSize()[1]-1)/2

    #生成内参矩阵
    intrinsic = np.array([[f/space_x,0        ,u,0],
                          [0        ,f/space_y,v,0],
                          [0        ,0        ,1,0]])

    """生成冠状面外参矩阵"""
    #读取放射源到patient plane的距离
    d = float(img.GetMetaData("0018|1111"))
    #读取二维影像旋转角度
    angle1 = float(img.GetMetaData("0018|1510"))*np.pi/180
    angle2 = float(img.GetMetaData("0018|1511"))*np.pi/180
    
    #生成外参矩阵这是一个从世界坐标系到图像坐标系的过程
    extrinsic1 = np.array([[1,0,0,0],
                           [0,-1,0,0],
                           [0,0,-1,d],
                           [0,0,0,1]])
    
    extrinsic2 = np.array([[ np.cos(angle1),0             ,np.sin(angle1),-d*np.sin(angle1)    ],
                           [ 0             ,1             ,0             , 0                   ],
                           [-np.sin(angle1),0             ,np.cos(angle1), d*(1-np.cos(angle1))],
                           [0,              0,0,              1]])
    
    extrinsic3 = np.array([[ 1             ,0             ,0              , 0                   ],
                           [ 0             ,np.cos(angle2),-np.sin(angle2), d*np.sin(angle2)    ],
                           [ 0             ,np.sin(angle2), np.cos(angle2), d*(1-np.cos(angle2))],
                           [ 0             ,0             ,0              , 1                   ]])
    
    extrinsic = extrinsic3@extrinsic2@extrinsic1
    return intrinsic, extrinsic, d, f, angle1, angle2

def image_plane_matrix(img:sitk.Image):
    space = img.GetSpacing()
    shape = img.GetSize()
    #读取焦距
    f = float(img.GetMetaData("0018|1110"))
    #读取放射源到patient plane的距离
    d = float(img.GetMetaData("0018|1111"))
    #读取二维影像旋转角度
    angle1 = float(img.GetMetaData("0018|1510"))*np.pi/180
    angle2 = float(img.GetMetaData("0018|1511"))*np.pi/180

    #这是一个从像素坐标系到世界坐标系的过程
    T1 = np.array([[1,0,0,-space[0]*(shape[0]-1)/2],
                   [0,1,0,-space[1]*(shape[1]-1)/2],
                   [0,0,1,f],
                   [0,0,0,1]])
    
    T2 = np.array([[1,0              ,0             ,0                ],
                   [0,np.cos(angle2) ,np.sin(angle2),-d*np.sin(angle2)],
                   [0,-np.sin(angle2),np.cos(angle2),d*(1-np.cos(angle2))],
                   [0,0              ,0             ,1]])
    
    T3 = np.array([[np.cos(angle1),0,-np.sin(angle1),d*np.sin(angle1)],
                   [0             ,1,0              ,0],
                   [np.sin(angle1),0,np.cos(angle1) ,d*(1-np.cos(angle1))],
                   [0             ,0,0              ,1]])
    T4 = np.array([[1,0,0,0],
                   [0,-1,0,0],
                   [0,0,-1,d],
                   [0,0,0,1]])
    return T4@T3@T2@T1


"""计算两个二维点集之间的相似度"""
def similarity(p_source, p_target):
    """
    计算p_souce各点到p_target各点的最短距离
    Input:
    p_source-[2,N1]
    p_target-[2,N2]
    Return:
    distance-[N1]
    """ 

    #p_source-[N1,2],p_target-[N2,2]
    p_source = p_source.T
    p_target = p_target.T

    min_ids = []
    distance_mask = np.zeros(p_target.shape[0])
    count = np.zeros(p_target.shape[0])

    for p in p_source:
        distance = np.sum((np.expand_dims(p,axis=0)-p_target)**2,axis=-1)
        distance = distance+distance_mask
        min_id = np.argmin(distance)
        min_ids.append(min_id)

        # count[min_id] = count[min_id]+1
        # if count[min_id] > 1:
        #     distance_mask[min_id] = 10000000

    return min_ids