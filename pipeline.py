import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from generate_line_segments import *
import sphere_mapping
lsd = cv2.createLineSegmentDetector(0)

def detect_ls(img, visualize=False):
    ls = lsd.detect(img)[0]

    if visualize:
        drawn_img = lsd.drawSegments(img,ls)
        cv2.imwrite('line_seg.png', drawn_img)
        cv2.imshow("LSD",drawn_img )
        cv2.waitKey(0)
    return ls.reshape(-1,4)

def normalize_lines_coord(img, ls):
    width = img.shape[1]
    height = img.shape[0]
    scale_w = np.maximum(width, height)
    scale_h = scale_w

    ls[:,0] -= width/2.0
    ls[:,1] -= height/2.0
    ls[:,2] -= width/2.0
    ls[:,3] -= height/2.0
    ls[:,0] /= (scale_w/2.0)
    ls[:,1] /= (scale_h/2.0)
    ls[:,2] /= (scale_w/2.0)
    ls[:,3] /= (scale_h/2.0)
    ls[:,1] *= -1
    ls[:,3] *= -1
    return ls

def run_cnn(img, model):

    model(img)
    pass


def project_back(cnn_grid, input_size, oringal_size, convert_original=True):

    [grid_H, grid_W] = cnn_grid.shape
    [input_H, input_W] = input_size
    vp_loc = np.vstack(np.nonzero(cnn_grid >= 0.5)).T # row, col -> y, x
    vp_loc = vp_loc.astype(np.float)
    grid_cell_H = input_H / grid_H 
    grid_cell_W = input_W / grid_W

    vp_loc[:,0] = vp_loc[:,0] * grid_cell_H + grid_cell_H/2
    vp_loc[:,1] = vp_loc[:,1] * grid_cell_W + grid_cell_W/2

    # plt.figure()
    # print(vp_loc.shape)
    # for i in range(vp_loc.shape[0]):
    #     plt.scatter(vp_loc[i,1], vp_loc[i,0])
    # plt.show(block = False)

    vp_loc[:,0] = -(vp_loc[:,0] / input_H * np.pi - np.pi/2) #y
    vp_loc[:,1] = (vp_loc[:,1] / input_W * np.pi - np.pi/2)  #x

    print(vp_loc.shape)
    plt.figure()
    print(vp_loc)
    for i in range(vp_loc.shape[0]):
        plt.scatter(vp_loc[i,1], vp_loc[i,0], color=color[i])
    plt.show(block = False)


    beta = vp_loc[:,0] 
    alpha = vp_loc[:,1]

    vp_point = np.vstack((np.sin(alpha)*np.cos(beta), np.sin(beta), np.cos(alpha)*np.cos(beta))).T
    vp_point = vp_point/ vp_point[:,2:3] #x, y, 1

    plt.figure()
    for i in range(vp_point.shape[0]):
        plt.scatter(vp_point[i,0], vp_point[i,1], color=color[i])
    plt.show(block = False)

    x = vp_point[:,0]
    y = vp_point[:,1]

    if convert_original:
        width,height = oringal_size
        scale_w = np.maximum(width, height)
        scale_h = scale_w

        x = x * scale_w/2
        y = y * -1
        y = y * scale_h/2

        x = x + width/2
        y = y + height/2

    return np.vstack((x,y)).T

def line_point_distance(lines, points):
    distance_matrix = np.zeros((lines.shape[0], points.shape[0]))

    for i in range(points.shape[0]):
        p = np.hstack((points[i,:], 1))
        distance = np.sum(lines * p, axis = 1)
        distance = np.abs(distance)/np.sqrt(lines[:,0] ** 2 + lines[:,1] ** 2)
        distance_matrix[:,i] = distance

    return distance_matrix

def compute_vp_from_lines(lines):
    A = lines[:,:2]
    b = - lines[:,2:3]

    solve = np.linalg.lstsq(A,b,rcond=None)[0]
    return solve

def em(lines, vps, total_iterations, vps_prior=False):

    new_vps = np.copy(vps)
    # sigma_k = np.ones(vps.shape[0])
    for i in range(total_iterations):
        #E-step
        distance_matrix = line_point_distance(lines, vps)

        #posterior = np.exp(-(distance_matrix ** 2))/(2 * sigma_k)
        assignment = np.argmin(distance_matrix, axis = 1)
        #print(assignment.shape)
        #M-step
        for vp in range(vps.shape[0]):
            assigned_lines = lines[assignment == vp, :]
            if assigned_lines.shape[0] > 2:
                recompute_vp = compute_vp_from_lines(assigned_lines)
                new_vps[vp,:] = recompute_vp.reshape(-1)

    return new_vps, assignment


def test_ground_truth():




    ls_p1s, ls_p2s, direction_assignment = generate_one_cluster(3, noise = False)
    print(direction_assignment)
    #print(ls_p1s)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    np.random.seed(9)

    for i in range(1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(ls_p1s.shape[0]):
            ax.plot([ls_p1s[i][0], ls_p2s[i][0]], [ls_p1s[i][1], ls_p2s[i][1]], [ls_p1s[i][2], ls_p2s[i][2]], color = color[direction_assignment[i]])
        ax.set_aspect('equal')
        R,t = generate_camera_extrinsics()
        K = generate_camera_intrinsic(t)
        ax.quiver(t[0,0], t[0,1], t[0,2], R[0,0], R[0,1], R[0,2], length=K[0,0], normalize=True, color = 'r')
        ax.quiver(t[0,0], t[0,1], t[0,2], R[1,0], R[1,1], R[1,2], length=K[0,0], normalize=True, color = 'g')
        ax.quiver(t[0,0], t[0,1], t[0,2], R[2,0], R[2,1], R[2,2], length=K[0,0], normalize=True, color = 'b')
        plt.show(block=False)
        

        valid_view, ls_p1s_2d, ls_p2s_2d, direction_assignment_2d, vps = proj2camera(ls_p1s, ls_p2s, direction_assignment, K, R, t)
        if not valid_view:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(ls_p1s_2d.shape[0]):
            ax.plot([ls_p1s_2d[i,0], ls_p2s_2d[i,0]], [ls_p1s_2d[i,1], ls_p2s_2d[i,1]],color = color[direction_assignment_2d[i]])
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect('equal')
        

        # for i in range(max_num_direction):
        #     if i in vps:
        #         ax.scatter(vps[i][0], vps[i][1],s = 50, marker = 'X' ,color= color[i])
        # ax.set_title('num_ls {}, num_vps {}'.format(ls_p1s_2d.shape[0], len(vps)))
        plt.show(block=False)





        num_ls = ls_p1s_2d.shape[0]
        leq = np.cross(np.hstack((ls_p1s_2d, np.ones((num_ls,1)))), np.hstack((ls_p2s_2d, np.ones((num_ls,1)))))

        vps = inverse_proj_vp(vps)



        pos_index = np.array(vps) * 64
        pos_index = pos_index.astype(np.int)
        grid_labels = np.zeros([64,64], dtype=np.int)
        grid_labels[pos_index[:, 0], pos_index[:, 1]] = 1
        

        plt.figure()

        plt.imshow(grid_labels)
        plt.show(block=False)

        vps_pred = project_back(grid_labels, [500,500], [2,2], False)
        print(vps_pred)
        plt.figure()
        plt.scatter(vps_pred[:,0], vps_pred[:,1])
        plt.show(block = False)

        new_vps, assignment = em(leq, vps_pred, 10)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(ls_p1s_2d.shape[0]):
            ax.plot([ls_p1s_2d[i,0], ls_p2s_2d[i,0]], [ls_p1s_2d[i,1], ls_p2s_2d[i,1]],color = color[assignment[i]])
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_aspect('equal')
        

        for i in range(new_vps.shape[0]):
            ax.scatter(new_vps[i,0], new_vps[i,1],s = 100, marker = 'o' ,color= color[i])
        for i in range(vps_pred.shape[0]):
            ax.scatter(vps_pred[i,0], vps_pred[i,1],s = 100, marker = 'x' ,color= color[i])
        ax.set_title('num_ls {}, num_vps {}'.format(ls_p1s_2d.shape[0], len(vps)))
        plt.show()


if __name__ == '__main__':


    test_ground_truth()
    # img = Image.open('mahatton.jpg')

    # img_gray = np.asarray(img.convert('L'))

    # ls = detect_ls(img_gray)
    # ls = normalize_lines_coord(img_gray, ls)

    # plt.figure()
    # for i in range(ls.shape[0]):
    #     plt.plot([ls[i,0],ls[i,2]], [ls[i,1], ls[i,3]]) 
    # plt.show()

    # ls_p1s_2d = ls[:,:2]
    # ls_p2s_2d = ls[:,2:]

    # num_ls = ls.shape[0]
    # leq = np.cross(np.hstack((ls_p1s_2d, np.ones((num_ls,1)))), np.hstack((ls_p2s_2d, np.ones((num_ls,1)))))
    # plt.figure()

    # img = sphere_mapping.sphere_line_plot(leq, 500)
    # img = Image.fromarray(img)
    # img.save('test.jpg')
    # plt.imshow(img)
    # plt.show()