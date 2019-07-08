#importing packages
import numpy as np


# linear kernel separately calculated in a function
# done separately since in future, other kernels will also be used
def linear_kernel(x1, x2):
    return np.dot(x1,x2)

def gaussian_kernel(x1, x2, sigma = 5.0):
    return np.exp( -(np.linalg.norm(x1 - x2) ** 2) / (2 * (sigma ** 2)) )
# np.linalg.norm uses frobeus norm method: [summation( abs(a[i][j]) ** 2) ] ** (1/2)
     
# ****************** SVM ************************
#==============================================================================
class SVM(object):
    
    def __init__(self, kernel=linear_kernel, C=None):
        
        self.kernel = kernel
        
        self.C = C
        if self.C is not None: self.C = float(self.C)
        
    # where the SVM will be trained
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        # the quadratic problem is:
        #   obj: min: (1/2)*x(T) * P * x + q(T) * x
        #   s.t.  G * x <= h ,
        #         A * x = b
        
        # our quadratic problem (hard-margin) is of form
        #   obj: min: (1/2)*x(T)*H*x - 1(T)*x   
        #   s.t.  -a <= 0
        #      y(T)*a = 0
        #
        # where H = y(i) * y(j) * (X(i).X(j)) and 'a' is alpha (lagrange mult.)
        #
        # how did we get that??
        # we already have, from the SVM maths,
        # the dual problem for hard-margin:
        # 
        # max[sum(a(i)) - (1/2) * sum(a(i) * a(j) * y(i) * y(j) * (X(i).X(j) ))]
        # 
  	# Now,
        # substitute H into it,
        #   obj: max: sum(a(i)) - (1/2)*a(T)*H*a
        #      s.t.   a(i) >= 0
        #  sum(a(i) * y(i)) = 0
        # Then,
        # convert sum to vector form (like, put all a(i) in a vector 'a')
        # and multiply everything by -1 to convert it to a minimization problem 
        
        # this will be used for calculating P
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j]) #can use diff kernels
        
        
        from cvxopt import matrix
        
        P = matrix(np.outer(y,y) * K)  # basically its: (Y.X)*(Y.X)T
        q = matrix(np.ones(n_samples) * -1) # a vector of -1s
        A = matrix(y, (1,n_samples)) # the label vector
        b = matrix(0.0) # single value matrix
        
        # other kernel's functionality will be added
        if self.C is None:
            G = matrix(np.diag(np.ones(n_samples) * -1))  # LHS constraint matrix
            h = matrix(np.zeros(n_samples))  # RHS of the constraints
        else:
            arr1 = np.diag(np.ones(n_samples) * -1)
            arr2 = np.diag(np.ones(n_samples))
            G = matrix(np.vstack((arr1, arr2)))
            
            arr1 = np.zeros(n_samples)
            arr2 = np.ones(n_samples) * self.C
            h = matrix(np.hstack((arr1, arr2)))
            
            #here the original optimization problem (soft-margin) is:
            #
            # max[sum(a(i)) - (1/2) * a(i) * a(j) * y(i) * y(j) * (X(i).X(j) )]
            #
            # s.t. 0 <= a(i) <= C
            #      sum( a(i) * y(i) ) = 0
            #
            # Similar to how it was done above, we convert it to:
            #
            # min[ ((1/2) * a(T) * H * a) - 1(T) * a]
            # s.t.
            #      -a <= 0 
            #       a <= C 
            # y(T) * a = 0
            
               
        # this class contains quadratic solver
        from cvxopt import solvers
        
        #some settings before applying solver
        solvers.options['show_progress'] = False  #if its left True then terminal
                                                  #will fillup with alpha values
                                                  #calculated in progression..
        # applying quadratic solver
        solution = solvers.qp(P,q,G,h,A,b)
        
        # Its solved!!!
        print("\n\nSolved !!!!!")
        #extracting all the corresponding alphas (lagrange multipliers)
        a = np.ravel(solution['x'])  #flattening function
        
        #extracting location of the support vectors
        sv = a > 1e-5 #returns a list of boolean values
                      #values per index depends upon the condition
                      # e.g. sv now is [False, True, False, False,....]
        print("\n\n{} out of {} are Support Vectors\n"
              .format(len(sv[sv==True]), n_samples))
        
        #Prepairing some parameters
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        
        print("\nThe Support Vectors are:\n", self.sv, "\n\n")
        
        
        #FINALLY WE CALCULATE THE OPTIMUM W and b
        
        if self.kernel == linear_kernel:
            self.w = ((y * a).T @ X).reshape(-1,1)
        else:
            self.w = None
        
        self.b = 0
        ind = np.arange(len(a))[sv]
        for i in range(len(self.a)):
            self.b += (self.sv_y)[i]
            self.b -= np.sum(self.a * self.sv_y * K[ind[i], sv])
        self.b /= len(self.a)

        
        print("Optimum weights =", self.w)
        print("\nOptimum b =", self.b)
        
               
        
    def project(self, x):
        if self.w is not None:
            return np.dot(x, self.w) + self.b
        else:
            y_pred = np.zeros(len(x))
            for i in range(len(x)):
                s = 0
                for ai, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += ai * sv_y * self.kernel(x[i], sv)
                y_pred[i] = s
            return y_pred + self.b
    
    
    def predict(self, x):
        return np.sign(self.project(x))
    
    def plot_contour(self, X1, X2):
    
        from matplotlib import pyplot as plt
        
        plt.figure(figsize=(8,6), dpi = 200)
        plt.plot(X1[:,0], X1[:,1], "ro", markersize=2)
        plt.plot(X2[:,0], X2[:,1], "bo", markersize=2)
        plt.scatter(self.sv[:,0], self.sv[:,1], s=10, c="g")
        
        x1, x2 = np.meshgrid(np.linspace(-10,10,50), np.linspace(-10,10,50))
        x = np.array( [[x1, x2] for x1,x2 in zip(np.ravel(x1), np.ravel(x2))])
        Z = self.project(x).reshape(x1.shape)

        plt.contour(x1, x2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        plt.contour(x1, x2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        plt.contour(x1, x2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        plt.show()

#==============================================================================
# ********** SVM *******************

if __name__ == "__main__":
    #generate linearly separable data
    def gen_lin_data():
        
        #distance between centers should be larger for larger amount of data
        mean1 = [5,0]
        mean2 = [0,7]
        
        cov = [[0.8,0.4],[0.4,0.8]]
        
        # generating normally distributed random numbers
        # for one class: +1
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        Y1 = np.ones(len(X1))
        
        # data for another class: -1
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        Y2 = np.ones(len(X2))*(-1)
        
        #merging the data of two separate classes into one
        X = np.concatenate((X1,X2))
        Y = np.concatenate((Y1,Y2))
        
        return X, Y
    
    def gen_lin_overlap_data():
        
        #distance between centers should be larger for larger amount of data
        mean1 = [2,0]
        mean2 = [0,2]
        
        cov = [[1.5,1.0],[1.0,1.5]]
        
        # generating normally distributed random numbers
        # for one class: +1
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        Y1 = np.ones(len(X1))
        
        # data for another class: -1
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        Y2 = np.ones(len(X2))*(-1)
        
        #merging the data of two separate classes into one
        X = np.concatenate((X1,X2))
        Y = np.concatenate((Y1,Y2))
        
        return X, Y
    
    def gen_nonlin_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        
        cov = [[1.0,0.8], [0.8, 1.0]]
        
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        
        X = np.concatenate((X1,X2))
        Y = np.concatenate((y1,y2))
        
        return X, Y
    
    #for testing linearly separable data    
    def test_hard():
        
        X, Y = gen_lin_data()
        
        from sklearn.model_selection import train_test_split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, train_size=0.9)
        
        svm = SVM()
        svm.fit(Xtrain, Ytrain)
         
        pred = svm.predict(Xtest)
        
        correct = np.sum(pred == Ytest.reshape(-1,1))
        
        print("\n{} out of {} predictions correct".format(correct, len(pred)))
    
       
        svm.plot_contour(Xtrain[Ytrain==1], Xtrain[Ytrain==-1])
    
    def test_soft():
        
        X, Y = gen_lin_overlap_data()
        
        from sklearn.model_selection import train_test_split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, train_size=0.9)
        
        svm = SVM(C = 1.0)
        svm.fit(Xtrain, Ytrain)
         
        pred = svm.predict(Xtest)
        
        correct = np.sum(pred == Ytest.reshape(-1,1))
    
        print("\n{} out of {} predictions correct".format(correct, len(pred)))
    
    
        svm.plot_contour(Xtrain[Ytrain==1], Xtrain[Ytrain==-1])
    
    def test_nonlin():
        
        X, Y = gen_nonlin_data()
        
        from sklearn.model_selection import train_test_split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, train_size=0.9)
        
        svm = SVM(kernel = gaussian_kernel)
        svm.fit(Xtrain, Ytrain)
        
        pred = svm.predict(Xtest).reshape(-1,1)
        
        correct = np.sum(pred == Ytest.reshape(-1,1))
        
        print("\n{} out of {} are correct".format(correct, len(pred)))
        
        svm.plot_contour(Xtrain[Ytrain==1], Xtrain[Ytrain==-1])
    
    #Execute
    test_nonlin()
    #test_soft()
#    test_hard()

    # =============================================================================
    # # plotting the whole result
    # def plot_margin(X1, X2, svm):
    #     
    #     def f(x, w, b, c=0):
    #         return (-w[0] * x - b + c)/w[1]
    #     
    #     #plotting the data points
    #     from matplotlib import pyplot as plt
    #     
    #     plt.figure(figsize=(8,6), dpi = 100)
    #     plt.plot(X1[:,0], X1[:,1], "ro", markersize=2) # 'ro' means red colored 'o' markers
    #     plt.plot(X2[:,0], X2[:,1], "bo", markersize=2) # 'b0' means blue colored 'o' markers
    #     plt.scatter(svm.sv[:,0], svm.sv[:,1], s=10, c="g") # support vectors
    #     
    #     #plotting the boundary and the margins
    #     a0 = -4 # these a0 and b0 are the x-axis value of the two points
    #     b0 = 8 # which will be used to draw the boundary line
    #     # the y-axis value will be wx + b for the boundary
    #     
    #     #the boundary line
    #     a1 = f(a0, svm.w, svm.b)
    #     b1 = f(b0, svm.w, svm.b)
    #     plt.plot([a0,b0], [a1,b1], "k") # 'k' is for black color
    #     
    #     #the margins
    #     a1 = f(a0, svm.w, svm.b, 1) # the last parameter creates a line parallel
    #     b1 = f(b0, svm.w, svm.b, 1) # to the boundary, with distance of +1
    #     plt.plot([a0,b0], [a1,b1], "k--") # dashed black line
    #     
    #     a1 = f(a0, svm.w, svm.b, -1) # the last paramter creates a line parallel
    #     b1 = f(b0, svm.w, svm.b, -1) # to the boundary, with distance of -1
    #     plt.plot([a0,b0], [a1,b1], "k--")
    #     
    #     plt.show()
    # =============================================================================
    
    
    
    
    
