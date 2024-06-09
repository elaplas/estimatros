import numpy as np
import sys

class Estimator:
    def __init__(self, X: np, Y: np, objective_function,  jac_func = None ,iter_max =100, allowed_error = 0.01, gama = 0.01):
        # residual errors
        self.err = None
        # uncertainty of estimated parameters
        self.cov = None
        # estimated parameters
        self.parameters = None
        # jacobian
        self.j = None
        # data points ( each row contains the data points yielding a measurement)
        self.X = X
        # measurements (each row contains the measurements resulted from one row of data points)
        self.Y = Y
        # objective function
        self.obj_func = objective_function
        # jacobian function
        self.jacobian_func = jac_func
        # learning rate
        self.gama = gama
        # maximum number of iterations
        self.iter_max = iter_max
        # allowed error ( norm of residual vector)
        self.allowed_err = allowed_error
        # control trust region
        self.control_trust_region = 0.01


    def set_intial_estimation(self, parameters):
        if type(parameters) is list:
            parameters = np.array(parameters).reshape(len(parameters), 0)
        self.parameters = parameters

    @staticmethod
    def get_slope(yield_from_left_side, yield_from_right_side, h):
        '''

        :param yield_from_left_side:  the result yielded from a point lies close on the left side of point
        :param yield_from_right_side: the result yielded from a point lies close on the right side of point
        :param h: step size
        :return: slope of tangent line
        '''
        return (yield_from_right_side - yield_from_left_side ) / (2*h)

    @staticmethod
    def get_Jacobian(objective_func, parameters: np, x = None):
        '''

        :param func: function that its gradient/jacobian is to be computed at the point "x"
        :param parameters: parameter vector (independable variables) of objective function
        :param x: data point
        :return: gradient at "parameter vector"
        '''

        # get machine epsilon
        eps = np.sqrt(np.finfo(float).eps)

        # handle the limited float point precision
        for i in range(len(parameters)):
            parameters[i, 0] = round(parameters[i,0], 2)

        # find a step size (delta) close to the parameter vector "parameters"
        h = [np.double(parameter*eps) if ( np.double(parameter) != 0 ) else eps for parameter in parameters]
        h = np.array(h).reshape(len(h), 1)
        # calculate a parameter point lies close to parameter vector "parameters" on the left side
        left_side = parameters - h
        # calculate a parameter point lies close to parameter vector "parameters" on the right side
        right_side = parameters + h

        # variable for storing the partial derivatives of one row of objective function
        g = None
        # calculate the slope of tangent line in all dimensions
        for i in range (len(parameters)):
            # form a parameter point on the left side of parameter vector that is moved by h_i only in one dimension/direction
            point_changed_only_in_one_direction_left_side = \
                [parameter if ( i!= count ) else left_side[count] for count, parameter in enumerate(parameters)]
            # form a parameter point on the right side of parameter vector that is moved by h_i only in one dimension/direction
            point_changed_only_in_one_direction_right_side = \
                [parameter if ( i!= count ) else right_side[count] for count, parameter in enumerate(parameters)]

            # wrap them to numpy array
            point_changed_only_in_one_direction_left_side = np.array( point_changed_only_in_one_direction_left_side)
            point_changed_only_in_one_direction_right_side = np.array(point_changed_only_in_one_direction_right_side)

            # calculate the result of objective function for the points moved only in one direction
            yielded_from_left_side = objective_func(point_changed_only_in_one_direction_left_side, x)
            yielded_from_right_side = objective_func(point_changed_only_in_one_direction_right_side, x)

            # handle scalars
            if type(yielded_from_left_side) != list and type(yielded_from_left_side) != np.ndarray:
                yielded_from_left_side = [yielded_from_left_side]
                yielded_from_right_side = [yielded_from_right_side]

            # determine the size of jacobian
            if i==0:
                g = np.zeros(( len(yielded_from_left_side), len(parameters)))

            # calculate the gradient for one column in the direction that the point is moved/changed
            for j in range (len(yielded_from_left_side)):
                # calculate the gradient for one element in the column
                g[j,i] = Estimator.get_slope(yielded_from_left_side[j], yielded_from_right_side[j], h[i])
        return g


    def compute_normal_equations(self, x: np, y: np):
        '''

        :param x: data point
        :param y: measurement
        :return: normal equations
        '''

        if self.parameters is None:
            print("set initial estimate of parameters at leas with zeros")
            return

        if self.jacobian_func is None:
            # If jacobian is not give, compute it numerically
            J = self.get_Jacobian(self.obj_func, self.parameters, x=x)
        else:
            # If jacobian is given, get it through the given function
            J = self.jacobian_func(self.parameters, x)
        JTJ = np.matmul(np.transpose(J),J)
        delta_y = y - (self.obj_func(self.parameters, x))
        JTY = np.matmul(np.transpose(J), delta_y)
        return JTJ, JTY, delta_y

    def estimate_gauss_newton(self, verbose = True):
        '''

        :param verbose: If true, the MSE and intermediate estimates are printed out
        :return: the estimate
        '''

        itr = 0
        while itr < self.iter_max:

            JTJ = None
            JTY = None
            Error = None
            for i in range (self.X.shape[0]):
                x = self.X[i].reshape(self.X.shape[1], 1)
                y = self.Y[i].reshape(self.Y.shape[1], 1)
                JTJ_i, JTY_i, delta_y = self.compute_normal_equations(x, y)

                # form JTJ and JTY if they don't exist
                if JTJ is None:
                    JTJ = np.zeros(JTJ_i.shape)
                    JTY = np.zeros(JTY_i.shape)
                    Error = np.zeros((delta_y.shape[0], 1))
                # accumulate normal equations of each set of data point
                JTJ = JTJ + JTJ_i
                JTY = JTY + JTY_i
                # accumulate residuals
                Error = Error + delta_y**2

            # calculate the mean of squared residuals
            self.err = Error/self.X.shape[0]

            if verbose:
                print("iteration ", itr, ":\n", " MSE: \n", self.err, "\n", " Estimate: \n", self.parameters, "\n")

            # handle the case JTJ is singular
            if np.linalg.cond(JTJ) > (1 / sys.float_info.epsilon):
                print("Hessian matrix, JTJ, is singular: Choose an initial guess closer to the estimate!")
                break

            # estimate the delta of parameters
            delta_parameters = np.matmul(np.linalg.inv(JTJ), JTY)

            # update the parameters
            self.parameters = self.parameters + delta_parameters

            # stop iterating if the MSE is acceptable
            if np.linalg.norm(self.err) <= (self.allowed_err*self.allowed_err):
                break

            itr += 1
        return self.parameters



    def estimate_levenberg_marquardt(self, verbose = True):
        '''

        :param verbose: If true, the MSE and intermediate estimates are printed out
        :return: the estimate
        '''

        itr = 0
        self.control_trust_region = 1.0

        while itr < self.iter_max:

            JTJ = None
            JTY = None
            Error = None
            for i in range(self.X.shape[0]):
                x = self.X[i].reshape(self.X.shape[1], 1)
                y = self.Y[i].reshape(self.Y.shape[1], 1)
                JTJ_i, JTY_i, delta_y = self.compute_normal_equations(x, y)

                # form JTJ and JTY if they don't exist
                if JTJ is None:
                    JTJ = np.zeros(JTJ_i.shape)
                    JTY = np.zeros(JTY_i.shape)
                    Error = np.zeros((delta_y.shape[0], 1))
                # accumulate normal equations of each set of data point
                JTJ = JTJ + JTJ_i
                JTY = JTY + JTY_i
                # accumulate residuals
                Error = Error + delta_y ** 2

            # calculate the mean of squared residuals of current estimate
            self.err = Error / self.X.shape[0]

            if verbose:
                print("iteration ", itr, ":\n", " MSE: \n", self.err, "\n", " Estimate: \n", self.parameters, "\n")

            # handle the case JTJ is singular
            if np.linalg.cond(JTJ) > (1 / sys.float_info.epsilon):
                print("Hessian matrix, JTJ, is singular: Choose an initial guess closer to the estimate!")
                break

            # estimate the delta of parameters
            delta_parameters_of_trust_region = np.matmul(np.linalg.inv(JTJ), JTY)
            # estimate the damped delta of parameters
            JTJ_smaller = self.control_trust_region * np.diag(np.diag(JTJ)) + JTJ
            delta_parameters_of_damped_trust_region = np.matmul(np.linalg.inv(JTJ_smaller), JTY)
            # estimate parameters based on trust region
            estimate_trust_region = self.parameters + delta_parameters_of_trust_region
            # estimate parameters based on damped trust region
            estimate_damped_trust_region = self.parameters + delta_parameters_of_damped_trust_region
            # compute errors
            error_trust_region = y - self.obj_func(estimate_trust_region , x)
            error_damped_trust_region = y - self.obj_func(estimate_damped_trust_region, x)


            # expand or shrink trust region depending on error and update accordingly
            if np.linalg.norm(error_damped_trust_region) < \
                    (np.linalg.norm(error_trust_region) * 0.5) :

                # shrink trust region
                self.control_trust_region *= 2
                # update the parameters
                self.parameters = self.parameters + delta_parameters_of_damped_trust_region
                print("I am............")

            else:
                # expand the trust region
                self.control_trust_region /= 3
                # update the parameters
                self.parameters = self.parameters + delta_parameters_of_trust_region

            # stop iterating if the MSE is acceptable
            if np.linalg.norm(self.err) <= (self.allowed_err * self.allowed_err):
                break

            itr += 1
        return self.parameters


    # TODO : Implement robust least square estimator( it is helpful when there are outliers )

    def robust_lm_estimate(self):
        pass


    def estimate_gradient_descent(self, verbose=False):

        itr = 0
        while itr < self.iter_max:
            delta_parameters_mean = np.zeros((len(self.parameters), 1))
            error_mean =  np.zeros((self.Y[0].shape[0], 1))
            for i in range(self.X.shape[0]):
                # get one data point and its corresponding measurement
                x = self.X[i].reshape(self.X.shape[1], 1)
                y = self.Y[i].reshape(self.Y.shape[1], 1)
                # get jacobian
                j = Estimator.get_Jacobian(self.obj_func, self.parameters, x)
                # get delta y
                error = self.obj_func(self.parameters, x) - y
                # compute gradient
                g = 2 * np.matmul(np.transpose(j), error)        

                # calculate the delta parameter
                delta_parameters_mean +=  self.gama * g
                # calculate residual for one point
                error_mean = error_mean + (error**2)
            # calculate the mean of maximum direction of change
            delta_parameters_mean = delta_parameters_mean / float(self.X.shape[0])
            # mean error
            error_mean = error_mean / self.X.shape[0]

            if verbose:
                print("iteration ", itr, ":\n", " MSE: \n", error_mean, "\n", " Estimate: \n", self.parameters, "\n")

            # update parameters
            self.parameters =  self.parameters - delta_parameters_mean

            if float(np.linalg.norm(error_mean)) <= (self.allowed_err**2):
                break
            itr += 1

        return self.parameters