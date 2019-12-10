import numpy as np
import matplotlib.pyplot as plt
import sphere_mapping
import coordinate_conversion
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import os
from collections import ChainMap
import json
import argparse

def close2zero(v):
    return np.isclose(np.linalg.norm(v), 0)  

def generate_random_direction():
    while True:
        v = np.random.uniform(-1,1,3)
        if not close2zero(v):
            return v

max_num_direction = 6
min_camera_distance = 2
max_camera_distance = 4
min_focal_length = 0.3
color = 'rgbcmyk'

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

min_noise_num = 10
max_noise_num = 20

final_min_num_ls = 20
final_success_rate = 0.7
final_min_num_vps = 1



def generate_one_cluster(num_direction):

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

    direction_assignment = []
    #Place clusters
    for d_idx, d in enumerate(directions):
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
                direction_assignment = direction_assignment + [d_idx] * num_ls
                #ls_p2s.append(ls_p2)

    #Place extra noise
    num_noise = np.random.randint(10, 20, 1)[0]
    n_p1 = np.random.uniform(-1, 1, (num_noise, 3))
    n_p2 = np.random.uniform(-1, 1, (num_noise, 3))


    ls_p1s = np.vstack((ls_p1s, n_p1))
    ls_p2s = np.vstack((ls_p2s, n_p2))

    direction_assignment = direction_assignment + [-1] * num_noise

    return ls_p1s, ls_p2s, np.array(direction_assignment)

def generate_camera_extrinsics():
    look_at_p = np.random.uniform(-0.3,0.3,3)
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

    f =  np.random.uniform(min_focal_length, np.linalg.norm(loc) * 0.6)
    #f = 1
    return np.array([[f, 0, 0], [0, f, 0],[0,0,1]])


def proj2camera(ls_p1s, ls_p2s, direction_assignment, K, R, t, visualize = False):

    ls_p1s_c = K.dot(R.T.dot((ls_p1s - t).T)).T
    ls_p1s_c = ls_p1s_c / ls_p1s_c[:,2:3]
    ls_p1s_c = ls_p1s_c[:,:2]

    ls_p2s_c = K.dot(R.T.dot((ls_p2s - t).T)).T
    ls_p2s_c = ls_p2s_c / ls_p2s_c[:,2:3]
    ls_p2s_c = ls_p2s_c[:,:2]


    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(ls_p1s_c.shape[0]):
            ax.plot([ls_p1s_c[i,0], ls_p2s_c[i,0]], [ls_p1s_c[i,1], ls_p2s_c[i,1]],color = color[direction_assignment[i]])
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect('equal')
        ax.set_title('num_ls {}'.format(ls_p1s_c.shape[0]))
        plt.show(block=False)

    #Filter out those lie outside of the box
    ls_p1s_c, ls_p2s_c, direction_assignment = clipping_ls_to_box(ls_p1s_c, ls_p2s_c, direction_assignment, np.array([[-1,1],[-1,1]]))


    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(ls_p1s_c.shape[0]):
            ax.plot([ls_p1s_c[i,0], ls_p2s_c[i,0]], [ls_p1s_c[i,1], ls_p2s_c[i,1]],color = color[direction_assignment[i]])
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect('equal')
        ax.set_title('num_ls {}'.format(ls_p1s_c.shape[0]))
        plt.show(block = False)

    valid_view = ls_p1s_c.shape[0] > 20
    vps = {}
    for i in range(max_num_direction):
        num_ls_per_d = np.count_nonzero(direction_assignment == i)
        indices = np.nonzero(direction_assignment == i)[0]
        if num_ls_per_d >= 2:
            two_choices = np.random.permutation(num_ls_per_d)[:2]
            p1 = ls_p1s_c[indices[two_choices[0]],:]
            p2 = ls_p2s_c[indices[two_choices[0]],:]

            p3 = ls_p1s_c[indices[two_choices[1]],:]
            p4 = ls_p2s_c[indices[two_choices[1]],:]

            leq1 = np.cross(np.hstack((p1, np.ones(1))), np.hstack((p2, np.ones(1))))
            leq2 = np.cross(np.hstack((p3, np.ones(1))), np.hstack((p4, np.ones(1))))

            vp = np.cross(leq1, leq2)
            vp = vp/vp[2]
            vp = vp[:2]

            vps[i] = vp

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(ls_p1s_c.shape[0]):
            ax.plot([ls_p1s_c[i,0], ls_p2s_c[i,0]], [ls_p1s_c[i,1], ls_p2s_c[i,1]],color = color[direction_assignment[i]])
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect('equal')
        

        for i in range(max_num_direction):
            if i in vps:
                ax.scatter(vps[i][0], vps[i][1], color= color[i])
        ax.set_title('num_ls {}, num_vps {}'.format(ls_p1s_c.shape[0], len(vps)))
        plt.show()

    valid_view = valid_view and len(vps) >= final_min_num_vps

    return valid_view, ls_p1s_c, ls_p2s_c, direction_assignment, vps

def inverse_proj_vp(vps):
    return_array = []
    for each in vps:
        vp = np.hstack((vps[each],1)) #Add 1 to make it homo
        vp = vp/np.linalg.norm(vp) #Normalize
        angle = coordinate_conversion.point_to_angle(vp)  #a, b -> x,y -> col, row

        a = (angle[0] - (-np.pi/2)) / np.pi #col
        b = 1 - (angle[1] - (-np.pi/2)) / np.pi #row

        grid_loc = [b,a]
        #row, col
        return_array.append(grid_loc)
    return return_array

def point_in_box(ls_p, box):
    return (ls_p[:,0] >= box[0,0]) & (ls_p[:,0] <= box[0,1]) & \
            (ls_p[:,1] >= box[1,0]) & (ls_p[:,1] <= box[1,1])

def point_out_box(ls_p, box):
    return ~point_in_box(ls_p, box)

def point_in_segs(p, a, b):

    ap = p - a
    bp = p - b
    return (np.sum(ap * bp, axis = 1) <= 0)



def clipping_ls_to_box(ls_p1, ls_p2, direction_assignment, box):
    """
    Clip line segments to the box -1 1
    box: [[-1,1], [-1,1]]
        [[x_lim][y_lim]]
    """
    #Only care if both point are out of the box
    point_mask = point_out_box(ls_p1, box) & point_out_box(ls_p2, box) 

    ls_p1_test = ls_p1[point_mask,:]
    ls_p2_test = ls_p2[point_mask,:]
    direction_assignment_test = direction_assignment[point_mask]

    num_test = ls_p1_test.shape[0]

    leq = np.cross(np.hstack((ls_p1_test, np.ones((num_test,1)))), np.hstack((ls_p2_test, np.ones((num_test,1))))) #line eq

    valid_intersect = np.zeros((num_test, 0)) # 0 0 0 1 -> has one intersect inside of the box
    box_eq = np.array([[0,1,box[0,0]], [0,1,box[0,1]], [1,0,box[1,0]], [1,0,box[1,1]]])

    for i in range(4):
        box_eq_now = box_eq[i,:]
        points = np.cross(leq, box_eq_now)
        points = points/points[:,2:3] #Normalize
        points = points[:,:2]
        in_box_mask = point_in_box(points, box) & point_in_segs(points, ls_p1_test, ls_p2_test)
        
        valid_intersect = np.hstack((valid_intersect, in_box_mask.reshape(-1,1)))


    score = np.sum(valid_intersect, axis = 1)
    good_point = score >= 2
    
    ls_p1_good = ls_p1_test[good_point,:]
    ls_p2_good = ls_p2_test[good_point,:]
    direction_assignment_good = direction_assignment_test[good_point]


    new_ls_p1 = np.vstack((ls_p1[~point_mask,:], ls_p1_good))
    new_ls_p2 = np.vstack((ls_p2[~point_mask,:], ls_p2_good))
    new_direction_assignment = np.hstack(([direction_assignment[~point_mask],direction_assignment_good]))

    return new_ls_p1, new_ls_p2, new_direction_assignment


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training image generate_random_direction")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--num_process",'-n', type=int, help="number of process", default = 32)
    parser.add_argument("--output_folder",'-o', type=str, help="output_folder", default='./output')
    parser.add_argument("--num_images_per_direction", '-nd', type = int, \
                        help="number of image per direction", default = 5000)
    args = parser.parse_args()
    if args.debug:
        ls_p1s, ls_p2s, direction_assignment = generate_one_cluster(6)
        print(direction_assignment)
        #print(ls_p1s)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        #print(ls_p1s[0][0])
        for i in range(ls_p1s.shape[0]):
            ax.plot([ls_p1s[i][0], ls_p2s[i][0]], [ls_p1s[i][1], ls_p2s[i][1]], [ls_p1s[i][2], ls_p2s[i][2]], color = color[direction_assignment[i]])
        ax.set_aspect('equal')
        # plt.show()


        for i in range(10):
            R,t = generate_camera_extrinsics()
            K = generate_camera_intrinsic(t)
            ax.quiver(t[0,0], t[0,1], t[0,2], R[0,0], R[0,1], R[0,2], length=K[0,0], normalize=True, color = 'r')
            ax.quiver(t[0,0], t[0,1], t[0,2], R[1,0], R[1,1], R[1,2], length=K[0,0], normalize=True, color = 'g')
            ax.quiver(t[0,0], t[0,1], t[0,2], R[2,0], R[2,1], R[2,2], length=K[0,0], normalize=True, color = 'b')
            plt.show(block=False)
            

            valid_view, ls_p1s_2d, ls_p2s_2d, direction_assignment_2d, vps = proj2camera(ls_p1s, ls_p2s, direction_assignment, K, R, t)

            num_ls = ls_p1s_2d.shape[0]
            leq = np.cross(np.hstack((ls_p1s_2d, np.ones((num_ls,1)))), np.hstack((ls_p2s_2d, np.ones((num_ls,1)))))
            plt.figure()

            img = sphere_mapping.sphere_line_plot(leq, 500)
            img = Image.fromarray(img)
            img.save('test.jpg')
            plt.imshow(img)
            plt.show()

            print(vps)
            print(inverse_proj_vp(vps))
    else:

        num_process = args.num_process
        num_images_per_direction = args.num_images_per_direction
        output_folder = args. output_folder
        
        def myfunc(*args):
            idx, direction, output_folder = args
            ls_p1s, ls_p2s, direction_assignment = generate_one_cluster(direction)
            complete = 0
            return_dict = {}

            while complete < 100:
                R,t = generate_camera_extrinsics()
                K = generate_camera_intrinsic(t)
                valid_view, ls_p1s_2d, ls_p2s_2d, direction_assignment_2d, vps = proj2camera(ls_p1s, ls_p2s, direction_assignment, K, R, t)
                if not valid_view:
                    continue
                num_ls = ls_p1s_2d.shape[0]
                leq = np.cross(np.hstack((ls_p1s_2d, np.ones((num_ls,1)))), np.hstack((ls_p2s_2d, np.ones((num_ls,1)))))

                img = sphere_mapping.sphere_line_plot(leq, 500)
                img = Image.fromarray(img)
                img_name = '{}_{}.jpg'.format(idx, complete)
                img.save(os.path.join(output_folder,img_name))
                return_dict[img_name] = inverse_proj_vp(vps)
                complete = complete + 1
            return return_dict


        def wrapMyFunc(*args):
            i,direction,output_folder = args
            return i, myfunc(i, direction,output_folder)

        def update(*args):
            # note: input comes from async `wrapMyFunc`
            (i, ans) = args[0]
            res[i] = ans  # put answer into correct index of result list
            pbar.update()
        num_point_sets = num_images_per_direction//100 #every point set has 100 view set up

        N = num_point_sets * 6 #total point set = num_point set * num of directions
        res = [None] * N #point sets result
        pbar = tqdm(total=N)
        print("NUM poinset:", num_point_sets)
        print("N:", N)
        pool = Pool(num_process)
        for i in range(N):
            direction = i//num_point_sets + 1 #get the number of direction
            pool.apply_async(wrapMyFunc, args=(i,direction,output_folder), callback=update)
        pool.close()
        pool.join()
        
        pbar.close()

        #Dump the json
        json_dict = {}
        json_dict['imgs'] = dict(ChainMap(*res))
        with open('label.json', 'w') as f:
           json.dump(json_dict, f, sort_keys=True, indent=4)
