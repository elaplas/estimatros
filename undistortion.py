
def undistort_point(p_x:float, p_y:float, calib_matrix:list, distortion_coeff_r:list, distortion_coeff_t:list):
    '''

    :param p_x: col coordinate of point of interest
    :param p_y: row coordinate of point of interest
    :param calib_matrix:  calibration matrix (3x3)
    :param distortion_coeff_r: radial distortion coefficients
    :param distortion_coeff_t: tangential distortion coefficients
    :return: undistored point as a tuple
    '''


    f_x, f_y, c_x, c_y,  = calib_matrix[0][0], calib_matrix[1][1], calib_matrix[0][2], calib_matrix[1][2]
    k1, k2, k3, k4, k5, k6 = distortion_coeff_r[0], distortion_coeff_r[1], distortion_coeff_r[2], \
                             distortion_coeff_r[3], distortion_coeff_r[4], distortion_coeff_r[5]
    t1, t2 = distortion_coeff_t[0], distortion_coeff_t[1]

    i_max = 5


    px_n = (p_x - c_x) / f_x
    py_n = (p_y - c_y) / f_y

    px_0 = px_n
    py_0 = py_n

    i = 0

    while( i < i_max ):

        #calculate r2, k_inv, delta_x and delta_y assuming that the solution is close
        #enough to the current point
        r2 = px_n**2 + py_n**2
        k_inv = (1 + k4*r2 + k5*(r2**2) + k6*(r2**3))/(1 + k1*r2 + k2*(r2**2) + k3*(r2**3))
        delta_x = 2*t1*px_n*py_n + t2*(r2+2*px_n**2)
        delta_y = 2*t2*px_n*py_n + t1*(r2+2*py_n**2)

        #update the solution
        px_n = (px_0 - delta_x) * k_inv
        py_n = (py_0 - delta_y) * k_inv

        i += 1

    return (px_n, py_n)



def distort_point(px:float, py:float, calib_matrix:list, distortion_coeff_r:list, distortion_coeff_t:list):
    '''
    Distort a point: This function is used only to test the correctness of estimator optimizing for the
    undistorted point

    :param p_x:  x coordinate of undistorted point
    :param p_y:  y coordinate of undistorted point
    :param calib_matrix:
    :param distortion_coeff_r: radial distortion coefficients
    :param distortion_coeff_t: tangential distortion coefficients
    :return: distored point as a tuple
    '''

    f_x, f_y, c_x, c_y, = calib_matrix[0][0], calib_matrix[1][1], calib_matrix[0][2], calib_matrix[1][2]
    k1, k2, k3, k4, k5, k6 = distortion_coeff_r[0], distortion_coeff_r[1], distortion_coeff_r[2], \
                             distortion_coeff_r[3], distortion_coeff_r[4], distortion_coeff_r[5]
    t1, t2 = distortion_coeff_t[0], distortion_coeff_t[1]

    r2 = px ** 2 + py ** 2
    k_x = (1 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3))
    k_y = (1 + k4 * r2 + k5 * (r2 ** 2) + k6 * (r2 ** 3))
    k = k_x / k_y
    delta_x = 2 * t1 * px * py + t2 * (r2 + 2 * (px ** 2))
    delta_y = 2 * t2 * px * py + t1 * (r2 + 2 * (py ** 2))

    # distort the point
    px_distorted = px * k + delta_x
    py_distorted = py * k + delta_y

    # scale the point
    px_distorted = px_distorted * f_x + c_x
    py_distorted = py_distorted * f_y + c_y

    return (px_distorted, py_distorted)




