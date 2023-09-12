import numpy as np
from scipy import sparse
from spatial_detrend.methods.util import vectorize

# Functions for solving for spatial systematics via variable projection gradient descent

# USAGE:
# 1) Initialize class with grid dimensions (_x, _y) and solver type (1 or 2) 
#    solve = Solver(_x, _y, 1)
# 2) Call generate_D with weight matrices Wx and Wy (use solver_weights to generate these if needed)
#    solve.generate_D(..)
# 3) Call solver with arguments, output is systematic coefficients C
#    C = solve.solve(...)
# 4) Estimate systematics for lightcurve data _Y_T
#    solve.est_syst(C, _Y_T)


def lp_norm(x, p):
    """
    Compute |x|_p^p

    Args:
    x : Input vector/matrix
    p : Norm 
    """
    return np.sum(np.abs(x)**p)


def _Tr(m,n):
    """
    Generate a transpose matrix _Tr which applied to vectorized matric vec(C), obtains vec(C^T)
    
    _Tr . vec(C) -> vec(C^T),
    where C is size mxn and C^T is size nxm
    vec()denotes vectorization of a matrix.
    
    Args:
    m, n : Dimensions of vectorized matrix C
        
    Returns:
    Transpose matrix _Tr of size (m*n) x (m*n).
    """

    tr = np.zeros((m*n,m*n))
    for i in range(m*n):
        tr[i][int(m*(i)-(n*m-1)*np.floor((i)/n))] = 1
    return tr

class Solver:
    def __init__(self, _x, _y, solver_type=1):
        self._x = _x
        self._y = _y
        self.solver_type = solver_type
        self.Dxx, self.Dyy = None, None

    def cost(self, _XT, C, p, k, alpha):
        """
        Calculates the current cost/penalty of estimated C according to penalty:: 
        """
        C = C.T
        CT_dag = np.linalg.inv(C.dot(C.T)).dot(C)
        P_CT = C.T.dot(CT_dag)
        function_cost = np.linalg.norm(_XT - P_CT.dot(_XT))
        C_T = C.T
        norm_vals = np.linalg.norm(C_T, axis=1)
        norm_C = (C_T / norm_vals[:,None]).T
        for g in range(k):
            Dx_ = self.Dxx.dot(norm_C[g])
            Dy_ = self.Dyy.dot(norm_C[g])
            function_cost += alpha*(lp_norm(Dx_, p) + lp_norm(Dy_, p))
        return function_cost

    def D_1(self, Wx, Wy):
        """
        Generate sparse difference operators Dx and Dy for matrix C of size (x, y).
        Operator Dx computes the the difference of coefficients C in the x direction.
        Operator Dy computes the difference of coefficients C in the y direction.
        This function returns operators which act on vec(C) and such that Dxx vec(C) = vec( Dx C )
        
        Args:
        x, y : Dimensions of vectorized matrix C
        Wx, Wy (optional) : Weight matrices, each element weights a corresponding difference calculation
            
        Returns:
        Dx, Dy : Difference operators for vectorized (C) which return vectorized differences
        """
        _1y = np.ones(self._y)
        _1y[self._y-1] = 0
        dy_s = np.array([-_1y, np.ones(self._y)])
        Dy_s = sparse.spdiags(dy_s, [0, 1], self._y, self._y)
        self.Dyy = sparse.kron(sparse.identity(self._x), Dy_s, format='csr')
        if Wy is not None:
            print (np.shape(Wy))
            Wy_diag = sparse.diags(Wy, 0, format='csr')
            self.Dyy = Wy_diag.dot(self.Dyy)
            
        _1x = np.ones(self._x)
        _1x[self._x-1] = 0
        dx_s =  np.array([-_1x, np.ones(self._x)])
        Dx_s = sparse.spdiags(dx_s, [0, 1], self._x, self._x, format='csr')
        self.Dxx = sparse.kron(Dx_s, sparse.identity(self._y), format='csr')
        if Wx is not None:
            Wx_diag = sparse.diags(Wx, 0, format='csr')
            self.Dxx = Wx_diag.dot(self.Dxx)

    def D_2(self, Wx, Wy):
        """
        Generate sparse difference operators Dxx and Dyy for matrix C of size (x, y).
        Operator Dx computes the the difference of coefficients C in the x direction.
        Operator Dy computes the difference of coefficients C in the y direction.
        This function returns operators which act on vec(C) and such that Dxx vec(C) = vec( Dx C )
        Dx shape ((x-1)*y, x*y)
        Dy shape (x(y-1), x*y)

        Args:
        x, y : Dimensions of vectorized matrix C (shape x*y)
        Wx, Wy (optional) : Weight matrices, each element weights a corresponding difference calculation
            
        Returns:
        Dxx, Dyy : Difference operators for vectorized (C) which return vectorized differences
        """
        # Sparse difference operator (to be applied to C_T)
        dy_s = np.array([-np.ones(self._y), np.ones(self._y)])
        Dy_s = sparse.spdiags(dy_s, [0, 1], self._y-1, self._y)
        self.Dyy = sparse.kron(sparse.identity(self._x), Dy_s, format='csr')
        if Wy is not None: 
            Wy_diag = sparse.diags(Wy, 0, format='csr')
            self.Dyy = Wy_diag.dot(self.Dyy)

        dx_s =  np.array([-np.ones(self._x), np.ones(self._x)])
        Dx_s = sparse.spdiags(dx_s, [0, 1], self._x-1, self._x, format='csr')
        self.Dxx = sparse.kron(Dx_s, sparse.identity(self._y), format='csr')
        if Wx is not None:
            Wx_diag = sparse.diags(Wx, 0, format='csr')
            self.Dxx = Wx_diag.dot(self.Dxx)

    def solver_1(self, k, _XT, C, p=2., alpha=.00001, eta=.1, beta=1e-16, niter=300, ls=1):
        """
        Gradient descent to minimize the variable projection cost/penalty with respect to C
        ls (bool) : if True perform a line search at each gradient step to obtain step size
        """

        XT_X = _XT.dot(_XT.T)
        C_T = C.T
        for i in range(niter):
            print (i)
            CT_dag = np.linalg.inv(C.dot(C_T)).dot(C)
            proj_CT = np.identity(self._x*self._y) - C_T.dot(CT_dag)
            A_ = - proj_CT.dot(XT_X).dot(CT_dag.T)
            B_ = np.zeros(np.shape(C_T))        
            norm_vals = np.linalg.norm(C_T, axis=1)
            norm_CT = np.diag(norm_vals**-1).dot(C_T)
            norm_C = norm_CT.T
            B_ = np.zeros(np.shape(C_T))
            grad_outer = np.zeros(np.shape(C_T))
            for g in range(k):
                Dx_ = self.Dxx.dot(norm_C[g])
                Dy_ = self.Dyy.dot(norm_C[g])
                wi = p / ( ( np.abs(Dx_)**2 + np.abs(Dy_)**2 + beta )**(1 - (p/2) ) )
                Wi = sparse.spdiags([wi], [0], self._x*self._y, self._x*self._y, format='csr') 
                grad_outer[:,g] = 2*alpha*(self.Dxx.T.dot(Wi).dot(self.Dxx) + self.Dyy.T.dot(Wi).dot(self.Dyy)).dot(norm_C[g])
            for j in range(self._x*self._y):
                grad_inner = (np.identity(k) - np.outer(norm_CT[j], norm_CT[j]))*(1/norm_vals[j])
                B_[j] = grad_outer[j].dot(grad_inner)
            grad = A_ + B_
            eta_ = eta
            if ls:
                while self.cost(_XT, C_T - eta_*grad, p, k, alpha) > self.cost(_XT, C_T, p, k, alpha) - (eta_/2)*np.linalg.norm(vectorize(grad)):
                    eta_ = .8*eta_
            if np.linalg.norm(eta_*grad) < 10e-5: break
            C_T -= eta_*grad
            C = C_T.T           
        return C

    def solver_2(self, k, _XT, C, p=2., alpha=.00001, eta=.1, beta=1e-16, niter=300, ls=1):
        """
        Gradient descent to minimize the variable projection cost/penalty with respect to C
        ls (bool) : if True perform a line search at each gradient step to obtain step size

        Calculate the cost as if |D_x C|_P + |D_y C|_P, instead of combined 2,p norm. 
        """
        XT_X = _XT.dot(_XT.T)
        C_T = C.T
        for i in range(niter):
            print (i)
            CT_dag = np.linalg.inv(C.dot(C_T)).dot(C)
            proj_CT = np.identity(self._x * self._y) - C_T.dot(CT_dag)
            A_ = - proj_CT.dot(XT_X).dot(CT_dag.T)
            B_ = np.zeros(np.shape(C_T))        
            norm_vals = np.linalg.norm(C_T, axis=1)
            norm_CT = np.diag(norm_vals**-1).dot(C_T)
            norm_C = norm_CT.T
            B_ = np.zeros(np.shape(C_T))
            grad_outer = np.zeros(np.shape(C_T))
            for g in range(k):
                Dx_ = self.Dxx.dot(norm_C[g])
                Dy_ = self.Dyy.dot(norm_C[g])
                wix = (p/2) / ( ( np.abs(Dx_)**2 + beta )**(1 - (p/2) ) )
                wiy = (p/2) / ( ( np.abs(Dy_)**2 + beta )**(1 - (p/2) ) )
                Wix = sparse.spdiags([wix], [0], (self._x-1)*self._y, (self._x-1)*self._y, format='csr') 
                Wiy = sparse.spdiags([wiy], [0], (self._y-1)*self._x, (self._y-1)*self._x, format='csr')
                grad_outer[:,g] = 2*alpha*(self.Dxx.T.dot(Wix).dot(self.Dxx) + self.Dyy.T.dot(Wiy).dot(self.Dyy)).dot(norm_C[g])
            for j in range(self._x*self._y):
                grad_inner = (np.identity(k) - np.outer(norm_CT[j], norm_CT[j]))*(1/norm_vals[j])
                B_[j] = grad_outer[j].dot(grad_inner)
            grad = A_ + B_
            eta_ = np.copy(eta)
            if ls:
                while self.cost(_XT, C_T - eta_*grad, p, k, alpha) > self.cost(_XT, C_T, p, k, alpha) - (eta_/2)*np.linalg.norm(vectorize(grad)):
                    eta_ = .8*eta_
            if np.linalg.norm(eta_*grad) < 10e-5: break
            print (eta_)
            C_T -= eta_*grad
            C = C_T.T           
        return C

    def generate_D(self, Wx = None, Wy = None):
        """
        Generate difference matrices Dx and Dy
        """
        if self.solver_type == 1:
            self.D_1(Wx, Wy)
        elif self.solver_type == 2:
            self.D_2(Wx, Wy)
        else:
            raise ValueError("Invalid solver_type!")

    def solve(self, k, _XT, C_init, p=2., alpha=.2, eta=.1, beta=1e-16, niter=30, ls=1):
        """
        Gradient descent updates of coefficients C with varial projection penalty

        Args:
        k - dimension
        _XT - lightcurve data of shape (_x*_y, N) where N is the length of a lightcurve
        C_init - initial coefficients of shape (k, _x*_y)
        p - Lp norm choice between [1, 2]
        alpha - weight of total variation prior,
        eta - gradient step size
        beta - offset for Huber l1 norm approximation
        niter - number of iterations (break condition also included)
        ls - whether to perform linesearch at each iteration for choice of step-size eta

        Returns:
        C - spatial coefficients of shape (k, _x*_y) - to obtain coefficients mapped to position: np.reshape(C[0], (_x, _y))
        """
        if self.Dxx is None and self.Dyy is None:
            self.generate_D()

        if self.solver_type == 1:
            C = self.solver_1(k, _XT, C_init, p, alpha, eta, beta, niter, ls)
        elif self.solver_type == 2:
            C = self.solver_2(k, _XT, C_init, p, alpha, eta, beta, niter, ls)
        else:
            raise ValueError("Invalid solver_type!")
        return C

    def est_syst(self, C, _XT):
        """
        Returns estimates sytematics for coefficients C and transposed data _XT under the spatial method
        To detrend lightcurves, -> _X_T -= est_syst(C, _XT)
        """
        CT_dag = np.linalg.inv(C.dot(C.T)).dot(C)
        P_CT = C.T.dot(CT_dag)
        return P_CT.dot(_XT)

