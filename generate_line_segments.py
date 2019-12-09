import numpy as np
import matplotlib.pyplot as plt


def close2zero(v):
    return np.isclose(np.linalg.norm(v), 0)  

def generate_random_direction():
    while True:
        v = np.random.uniform(-1,1,3)
        if not close2zero(v):
            return v


min_camera_distance = 2
max_camera_distance = 4
min_focal_length = 0.5


def generate_one_cluster():


    num_direction = 5
    total_direction = []

    min_clusters = 2
    max_clusters = 6
    min_ls = 1
    max_ls = 10
    displacement_std = 0.3 

    colinear_prob = 0.5
    min_colinear_ls = 1
    max_colinear_ls = 3
    colinear_displacement_std = 1
    colinear_noise_std = 0.001

    min_ls_length = 0.1
    max_ls_length = 1

    #Generate direction
    v1 = generate_random_direction()
 
    #First direction
    v1 = v1/np.linalg.norm(v1)

    #Second direction
    while True:
        v2 = generate_random_direction()
        v2 = np.cross(v1, v2)
        if not close2zero(v2):
            v2 = v2/np.linalg.norm(v2)
            break
    
    #Third direction
    v3 = np.cross(v1, v2)
    v3 = v3/np.linalg.norm(v3)

    directions = [v1, v2, v3]

    for i in range(num_direction - len(directions)):

        indices = np.random.permutation(len(directions))[:2]

        coeff = np.random.uniform(-1, 1, 2)
        while True:
            new_v = np.dot(np.array(directions)[indices,:].T, np.reshape(coeff, (2, 1)))
            if not close2zero(new_v):
                new_v = new_v/np.linalg.norm(new_v)
                break
        directions.append(np.reshape(new_v, -1))

    ls_p1s = np.zeros((0,3))
    ls_p2s = np.zeros((0,3))

    #Place clusters
    for d in directions:
        num_clusters = np.random.randint(min_clusters, max_clusters, 1)[0]
        cluster_center = np.random.uniform(-1, 1, (num_clusters, 3))

        for c in range(num_clusters):
            for t in ['colinear', 'parallel']:
                num_ls = np.random.randint(min_ls, max_ls, 1)[0]
                if t == 'colinear':
                    if np.random.rand() > colinear_prob:
                        continue
                    displacement = np.random.normal(0,colinear_displacement_std, (num_ls,1)) * d
                    displacement = displacement + np.random.normal(0,colinear_noise_std, (num_ls,3))

                elif t == 'parallel':
                    displacement = np.random.normal(0,displacement_std, (num_ls,3))
                
                ls_centers = cluster_center[c,:] + displacement

                ls_length = np.random.uniform(min_ls_length, max_ls_length, (num_ls, 1))

                ls_p1 = ls_centers - d * ls_length/2
                ls_p2 = ls_centers + d * ls_length/2

                ls_p1s = np.vstack((ls_p1s, ls_p1))
                ls_p2s = np.vstack((ls_p2s, ls_p2))
                #ls_p2s.append(ls_p2)

    #Place extra noise


    return ls_p1s, ls_p2s

def generate_camera_extrinsics():
    look_at_p = np.random.uniform(-1,1,3)
    camera_radius = np.random.uniform(min_camera_distance,max_camera_distance,1)[0]

    while True:
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)

        camera_locs = vec * camera_radius
        v_z = look_at_p - camera_locs
        if not close2zero(v_z):
            v_z = v_z/np.linalg.norm(v_z)
            break


    while True:
        v_y = generate_random_direction()

        v_x = np.cross(v_z, v_y)
        if not close2zero(v_x):
            v_x = v_x/np.linalg.norm(v_x)
            v_y = np.cross(v_x, v_z)
            v_y = v_y/np.linalg.norm(v_y)
            break
    R = np.vstack([v_x, v_y, v_z])
    t = np.reshape(camera_locs, (1,3))
    return R, t


def generate_camera_intrinsic(loc):

    f =  np.random.uniform(min_focal_length, np.linalg.norm(loc) * 0.8)
    return np.array([[f, 0, 0], [0, f, 0],[0,0,1]])


def proj2camera(ls_p1s, ls_p2s, K, R, t):
    print(K)
    print(t)
    ls_p1s_c = K.dot(R.T.dot((ls_p1s - t).T))
    ls_p1s_c = ls_p1s_c / ls_p1s_c[:,2:3]


    ls_p2s_c = K.dot(R.T.dot((ls_p2s - t).T))
    ls_p2s_c = ls_p2s_c / ls_p2s_c[:,2:3]

    ls_p1s_c = ls_p1s_c.T
    ls_p2s_c = ls_p2s_c.T

    plt.figure()

    for i in range(ls_p1s_c.shape[0]):
        plt.plot([ls_p1s_c[i,0], ls_p2s_c[i,0]], [ls_p1s_c[i,1], ls_p2s_c[i,1]])
    plt.show()

def proj2sphere():
    pass




ls_p1s, ls_p2s = generate_one_cluster()
#print(ls_p1s)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#print(ls_p1s[0][0])
for i in range(ls_p1s.shape[0]):
    ax.plot([ls_p1s[i][0], ls_p2s[i][0]], [ls_p1s[i][1], ls_p2s[i][1]], [ls_p1s[i][2], ls_p2s[i][2]])
ax.set_aspect('equal')
# plt.show()


for i in range(1):
    R,t = generate_camera_extrinsics()
    # ax.quiver(t[0], t[1], t[2], R[0,0], R[0,1], R[0,2], length=0.2, normalize=True, color = 'r')
    # ax.quiver(t[0], t[1], t[2], R[1,0], R[1,1], R[1,2], length=0.2, normalize=True, color = 'g')
    # ax.quiver(t[0], t[1], t[2], R[2,0], R[2,1], R[2,2], length=0.2, normalize=True, color = 'b')

    K = generate_camera_intrinsic(t)

    proj2camera(ls_p1s, ls_p2s, K, R, t)



# ax.set_aspect('equal')
# plt.show()

