import numpy as np



def projection_error(parameters, data_point):

    '''

    Projection error function

    :param parameters:   parameters to be estimated (three translations and three rotations)
    :param data_point:   observations ( control 3D points in WCS)
    :return:             measurements ( image coordinates on z-plane )

    '''

    # 3D point in WCS
    x = np.double(data_point[0])
    y = np.double(data_point[1])
    z = np.double(data_point[2])

    # translation from WCS -> CCS
    t_x, t_y, t_z = np.double(parameters[0]), np.double(parameters[1]), np.double(parameters[2])
    # rotation vector from WCS -> CCS
    r_x, r_y, r_z = np.double(parameters[3]), np.double(parameters[4]), np.double(parameters[5])

    # compute rotation from WCS -> CCS
    R = get_rotation_matrix(r_x, r_y, r_z);
    # form translation vector
    t = np.array([[t_x], [t_y], [t_z]])
    # form transformation matrix from WCS -> CCS
    T = np.concatenate((R, t ), axis=1)
    # form data point vector
    point_3d = np.array([[x], [y], [z], [1]])
    # transform point from WCS -> CCS
    point_3d_in_CCS = np.matmul(T, point_3d)
    # Project 3D point into z plane (normalized image points)
    predicted_px = point_3d_in_CCS[0,0] / point_3d_in_CCS[2,0]
    predicted_py = point_3d_in_CCS[1,0] / point_3d_in_CCS[2,0]

    predicted_measurement = np.array([[predicted_px], [predicted_py]])

    return predicted_measurement



def get_rotation_matrix(r_x:float, r_y:float, r_z:float):

    # rotation about z-axis
    R_z = np.array([ [np.cos(r_z),  - np.sin(r_z),   0],
                     [np.sin(r_z),    np.cos(r_z),   0],
                     [0,        0,                   1]])

    # rotation about y-axis
    R_y = np.array([[np.cos(r_y),   0, np.sin(r_y)],
                    [0,             1,           0],
                    [- np.sin(r_y), 0, np.cos(r_y)]])

    # rotation about x-axis
    R_x = np.array([
                    [1,           0,             0],
                    [0, np.cos(r_x), - np.sin(r_x)],
                    [0, np.sin(r_x),  np.cos(r_x) ],
                    ])

    return np.matmul(R_z, np.matmul(R_y, R_x))




def get_jacobian_of_projection_error(parameters, data_point):
    '''
    Jacobina of projection error function

    :param parameters:   parameters to be estimated (three translations and three rotations)
    :param data_point:   observations ( control 3D points in WCS)
    :return:             measurements ( image coordinates on z-plane )

    '''

    # 3D point in WCS
    x = np.double(data_point[0])
    y = np.double(data_point[1])
    z = np.double(data_point[2])

    # translation from WCS -> CCS
    t_x, t_y, t_z = np.double(parameters[0]), np.double(parameters[1]), np.double(parameters[2])
    # rotation vector from WCS -> CCS
    r_x, r_y, r_z = np.double(parameters[3]), np.double(parameters[4]), np.double(parameters[5])


    # gradients w.r.t t_x
    j_1 = np.array([
                    1/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z),
	                0
    ])

    # gradients w.r.t t_y
    j_2 = np.array([
        0,
        1/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)
    ])

    # gradients w.r.t t_z
    j_3 = np.array([
    -((np.sin(r_x)*np.sin(r_z)+np.cos(r_x)*np.sin(r_y)*np.cos(r_z))*z+(np.sin(r_x)*np.sin(r_y)*np.cos(r_z) \
    -np.cos(r_x)*np.sin(r_z))*y+np.cos(r_y)*np.cos(r_z)*x+t_x)/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)* \
    np.cos(r_y)*y-np.sin(r_y)*x+t_z)**2,

	-((np.cos(r_x)*np.sin(r_y)*np.sin(r_z)-np.sin(r_x)*np.cos(r_z))*z+(np.sin(r_x)*np.sin(r_y)* \
    np.sin(r_z)+np.cos(r_x)*np.cos(r_z))*y+np.cos(r_y)*np.sin(r_z)*x+t_y)/(np.cos(r_x)*np.cos(r_y)* \
    z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)**2
	])

    # gradients w.r.t r_x
    j_4 = np.array([
	((np.cos(r_x)*np.sin(r_z)-np.sin(r_x)*np.sin(r_y)*np.cos(r_z))*z+(np.sin(r_x)*np.sin(r_z)+np.cos(r_x)* \
    np.sin(r_y)*np.cos(r_z))*y)/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)-\
    ((np.cos(r_x)*np.cos(r_y)*y-np.sin(r_x)*np.cos(r_y)*z)*((np.sin(r_x)*np.sin(r_z)+np.cos(r_x)*np.sin(r_y)* \
    np.cos(r_z))*z+(np.sin(r_x)*np.sin(r_y)*np.cos(r_z)-np.cos(r_x)*np.sin(r_z))*y+np.cos(r_y)*np.cos(r_z)* \
    x+t_x))/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)**2,

	((-np.sin(r_x)*np.sin(r_y)*np.sin(r_z)-np.cos(r_x)*np.cos(r_z))*z+(np.cos(r_x)*np.sin(r_y)*np.sin(r_z)- \
    np.sin(r_x)*np.cos(r_z))*y)/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)- \
    ((np.cos(r_x)*np.cos(r_y)*y-np.sin(r_x)*np.cos(r_y)*z)*((np.cos(r_x)*np.sin(r_y)*np.sin(r_z)-np.sin(r_x)* \
    np.cos(r_z))*z+(np.sin(r_x)*np.sin(r_y)*np.sin(r_z)+np.cos(r_x)*np.cos(r_z))*y+np.cos(r_y)*np.sin(r_z)* \
    x+t_y))/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)**2
	])

    # gradients w.r.t r_y
    j_5 = np.array([
    (np.cos(r_x)*np.cos(r_y)*np.cos(r_z)*z+np.sin(r_x)*np.cos(r_y)*np.cos(r_z)*y-np.sin(r_y)*np.cos(r_z) \
     *x)/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)-((-np.cos(r_x)*np.sin(r_y)* \
    z-np.sin(r_x)*np.sin(r_y)*y-np.cos(r_y)*x)*((np.sin(r_x)*np.sin(r_z)+np.cos(r_x)*np.sin(r_y)*np.cos(r_z))* \
    z+(np.sin(r_x)*np.sin(r_y)*np.cos(r_z)-np.cos(r_x)*np.sin(r_z))*y+np.cos(r_y)*np.cos(r_z)*x+t_x))/(np.cos(r_x)* \
    np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)**2,

	(np.cos(r_x)*np.cos(r_y)*np.sin(r_z)*z+np.sin(r_x)*np.cos(r_y)*np.sin(r_z)*y-np.sin(r_y)*np.sin(r_z)* \
    x)/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)-((-np.cos(r_x)*np.sin(r_y)* \
    z-np.sin(r_x)*np.sin(r_y)*y-np.cos(r_y)*x)*((np.cos(r_x)*np.sin(r_y)*np.sin(r_z)-np.sin(r_x)*np.cos(r_z))* \
    z+(np.sin(r_x)*np.sin(r_y)*np.sin(r_z)+np.cos(r_x)*np.cos(r_z))*y+np.cos(r_y)*np.sin(r_z)*x+t_y))/(np.cos(r_x)* \
    np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)*y-np.sin(r_y)*x+t_z)**2
	])

    # gradients w.r.t r_z
    j_6 = np.array([
	((np.sin(r_x)*np.cos(r_z)-np.cos(r_x)*np.sin(r_y)*np.sin(r_z))*z+(-np.sin(r_x)*np.sin(r_y)*np.sin(r_z)- \
    np.cos(r_x)*np.cos(r_z))*y-np.cos(r_y)*np.sin(r_z)*x)/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)* \
    y-np.sin(r_y)*x+t_z),

	((np.sin(r_x)*np.sin(r_z)+np.cos(r_x)*np.sin(r_y)*np.cos(r_z))*z+(np.sin(r_x)*np.sin(r_y)*np.cos(r_z)- \
    np.cos(r_x)*np.sin(r_z))*y+np.cos(r_y)*np.cos(r_z)*x)/(np.cos(r_x)*np.cos(r_y)*z+np.sin(r_x)*np.cos(r_y)* \
    y-np.sin(r_y)*x+t_z)
    ])

    # form jacobian matrix
    J = np.array([j_1, j_2, j_3, j_4, j_5, j_6])

    return np.transpose(J)



def test_cost_fun(parameters, data_point):
    '''

    this function is only used to test the correctness of some implemented algorithms

    :param parameters: parameters to be estimated
    :param data_point: observations
    :return:
    '''

    a = parameters[0]
    b = parameters[1]

    x = data_point[0]
    y = data_point[1]

    cost = (a*x) + (b*y)

    return np.array([cost])