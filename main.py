import numpy as np
from objective_function import test_cost_fun, projection_error, get_jacobian_of_projection_error
from estimator import Estimator
from undistortion import distort_point, undistort_point


def compare_float(a, b, tolerance = 0.1):

    if abs(a-b) <= tolerance:
        print('passed: ', a, ' against ' ,  b, "\n" )
    else:
        print('failed: ', a, ' against ' , b, "\n")


def compare_int(a, b):

    if a == b:
        print("passed: ", a, " against " , b, "\n" )
    else:
        print("failed: ", a, " against ", b, "\n")


def test_jacobian():

    print("\n", ".........start test 1: numerical jacobian vs parameterical jacobian...........")

    cam_pose = np.array([[0.0],
                         [0.0],
                         [0.0],
                         [0.5],
                         [0.0],
                         [0.9]])

    point3d = np.array([[2], [2], [2]])

    # Compute jacobian numerically
    J_numerical = get_jacobian_of_projection_error(cam_pose, point3d)

    # Calculate jacobian parametrically
    J2_parameterical = Estimator.get_Jacobian(projection_error, cam_pose, point3d)


    compare_int(J_numerical.shape[0], J2_parameterical.shape[0])
    compare_int(J_numerical.shape[1], J2_parameterical.shape[1])

    compare_float(J_numerical[0, 0], J2_parameterical[0, 0])
    compare_float(J_numerical[1, 0], J2_parameterical[1, 0])
    compare_float(J_numerical[0, 1], J2_parameterical[0, 1])
    compare_float(J_numerical[1, 1], J2_parameterical[1, 1])
    compare_float(J_numerical[0, 2], J2_parameterical[0, 2])
    compare_float(J_numerical[1, 2], J2_parameterical[1, 2])
    compare_float(J_numerical[0, 3], J2_parameterical[0, 3])
    compare_float(J_numerical[1, 3], J2_parameterical[1, 3])
    compare_float(J_numerical[0, 4], J2_parameterical[0, 4])
    compare_float(J_numerical[1, 4], J2_parameterical[1, 4])
    compare_float(J_numerical[0, 5], J2_parameterical[0, 5])
    compare_float(J_numerical[1, 5], J2_parameterical[1, 5])

    print("\n", ".........end test 1: numerical jacobian vs parameterical jacobian...........")

def test_gauss_newton():

    print("\n", ".........start test 2: check correctness of gauss_newton estimator...........")

    data_points = np.array([[1, 2],
                            [1, 1],
                            [1.5, 3],
                            [3, 1]])

    measurements = np.array([ [2],
                              [3],
                              [0.5],
                              [1.5]])

    estimator = Estimator(data_points, measurements, test_cost_fun )
    initial_guess = np.array([[0],[0]])
    estimator.set_intial_estimation(initial_guess)

    #estimate = estimator.estimate_gauss_newton(verbose=False)
    estimate = estimator.estimate_gauss_newton(verbose=False)
    expected_res = np.linalg.lstsq(data_points, measurements, rcond=None)


    compare_float(estimate[0, 0], expected_res[0][0, 0])
    compare_float(estimate[1, 0], expected_res[0][1, 0])

    print(".........end test 2: check correctness of gauss_newton estimator.........." + "\n")


def test_lm():

    print("\n", ".........start test 3: check correctness of lm estimator estimator...........")

    data_points = np.array([[1, 2],
                            [1, 1],
                            [1.5, 3],
                            [3, 1]])

    measurements = np.array([ [2],
                              [3],
                              [0.5],
                              [1.5]])

    estimator = Estimator(data_points, measurements, test_cost_fun )
    initial_guess = np.array([[0],[0]])
    estimator.set_intial_estimation(initial_guess)

    #estimate = estimator.estimate_gauss_newton(verbose=False)
    estimate = estimator.estimate_levenberg_marquardt(verbose=False)
    expected_res = np.linalg.lstsq(data_points, measurements, rcond=None)


    compare_float(estimate[0, 0], expected_res[0][0, 0])
    compare_float(estimate[1, 0], expected_res[0][1, 0])

    print(".........end test 3: check correctness of lm estimator........." + "\n")


def test_undistortaion():


    print("\n", ".........start test 4: check correctness of undistortion estimator...........")

    undis_point = [2,3]

    radial_coff = [
        -0.0715295127,
        -0.000106265925,
        -0.00425297258,
        0.121668948,
        -0.0175844949,
        -0.00788054176]

    tangential_coff = [
        -0.000488930189,
        -0.000236289772
    ]

    px = 653.8606098990497
    py = 505.54595809472005
    fx = 1442.3335964529872
    fy = 1442.6850199153005

    cam_matrix = [[fx, 0.0, px], [.00, fy, py], [0.0,0.0,1.0]]

    # distort the point
    dis_point = distort_point(undis_point[0], undis_point[1], cam_matrix, radial_coff, tangential_coff)

    # undistort the point
    estimate = undistort_point(dis_point[0], dis_point[1], cam_matrix, radial_coff, tangential_coff)


    compare_float(estimate[0], undis_point[0])
    compare_float(estimate[1], undis_point[1])

    print(".........end test 4: check correctness of undistortion estimator.........." + "\n")

def test_estimate_camera_pose():

    print("..........start test 5: estimate camera pose given measurements..........." + "\n")

    # Camera pose to be estimate
    cam_pose_1 = np.array([[0.5],
                        [0.5],
                        [0.5],
                        [0.9],
                        [0.3],
                        [1.1]])

    # 3D points in WCS (control points)
    points_3d = np.array([[10, 10, 2],
                         [10, 150, 2],
                         [100, 150, 2],
                         [100, 10, 2]])

    # Generate synthetic camera measurements ( pixel coordinates)
    measurements_in_pos1 = np.zeros((points_3d.shape[0], 2))
    for i in range(len(points_3d)):
        point_3d =  points_3d[i].reshape( points_3d.shape[1], 1)
        projected_point = projection_error(cam_pose_1, point_3d)
        projected_point = np.transpose(projected_point)
        measurements_in_pos1[i, :] = projected_point

    # Instantiate "Estimator" class when the jacobian of objective function is not given
    estimator_without_jacobian = Estimator(points_3d, measurements_in_pos1, projection_error)

    # Instantiate "Estimator" class when the jacobian of objective function is not given
    estimator_with_jacobian = Estimator(points_3d, measurements_in_pos1, projection_error, \
                                        jac_func = get_jacobian_of_projection_error, gama=.005)

    # Initial guess of the estimate
    initial_guess = np.array([[2.0], [1.0], [3.0], [0.2], [0.1], [0.1]])

    # Set Initial guess
    estimator_without_jacobian.set_intial_estimation(initial_guess)
    estimator_with_jacobian.set_intial_estimation(initial_guess)

    # Estimate given jacobian
    estimate_with_jacobian = estimator_with_jacobian.estimate_gradient_descent(verbose=False)
    print("estimate given jacobina: \n", estimate_with_jacobian, "\n")
    estimate_without_jacobian = estimator_without_jacobian.estimate_levenberg_marquardt(verbose=False)
    print("estimate without jacobina: \n", estimate_without_jacobian, "\n")

    compare_float(estimate_with_jacobian[0, 0], cam_pose_1[0, 0])
    compare_float(estimate_with_jacobian[1, 0], cam_pose_1[1, 0])
    compare_float(estimate_with_jacobian[2, 0], cam_pose_1[2, 0])
    compare_float(estimate_with_jacobian[3, 0], cam_pose_1[3, 0])
    compare_float(estimate_with_jacobian[4, 0], cam_pose_1[4, 0])
    compare_float(estimate_with_jacobian[5, 0], cam_pose_1[5, 0])



    print("\n", "..........end test 5: estimate camera pose given measurements...........")


if __name__ == "__main__":

   test_jacobian()

   test_gauss_newton()

   test_lm()

   test_undistortaion()

   test_estimate_camera_pose()





