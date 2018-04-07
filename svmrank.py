# -*- coding: utf-8 -*-
"""
(Thin) Wrapper for SVM^{rank}
Created on Tue Mar 06 12:14:10 2018
@author: Dr. Fayyaz Minhas (afsar at pieas.edu dot pk)
"""
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import os
import uuid
import numpy as np
from shutil import copyfile
from glob import glob
from time import time

__svm_rank_path__ = 'svmrank' #path to the folder containing svm_rank_classify and svm_rank_learn

class SVMrank:
    """
    (Thin) Wrapper for SVM^{rank}: https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
    The following two module level variables should point to svm_rank_learn and svm_rank_classify executables
    You cand download the binaries or the source code from the SVM^{rank} URL and use it.
    __svm_fit_path__
    __svm_fit_path__
    
    General Options:
             -v [0..3]   -> verbosity level (default 1)
             -y [0..3]   -> verbosity level for svm_light (default 0)
    Learning Options:
             -c float    -> C: trade-off between training error
                            and margin (default 0.01)
             -p [1,2]    -> L-norm to use for slack variables. Use 1 for L1-norm,
                            use 2 for squared slacks. (default 1)
             -o [1,2]    -> Rescaling method to use for loss.
                            1: slack rescaling
                            2: margin rescaling
                            (default 2)
             -l [0..]    -> Loss function to use.
                            0: zero/one loss
                            ?: see below in application specific options
                            (default 1)
    Optimization Options (see [2][5]):
             -w [0,..,9] -> choice of structural learning algorithm (default 3):
                            0: n-slack algorithm described in [2]
                            1: n-slack algorithm with shrinking heuristic
                            2: 1-slack algorithm (primal) described in [5]
                            3: 1-slack algorithm (dual) described in [5]
                            4: 1-slack algorithm (dual) with constraint cache [5]
                            9: custom algorithm in svm_struct_learn_custom.c
             -e float    -> epsilon: allow that tolerance for termination
                            criterion (default 0.001000)
             -k [1..]    -> number of new constraints to accumulate before
                            recomputing the QP solution (default 100)
                            (-w 0 and 1 only)
             -f [5..]    -> number of constraints to cache for each example
                            (default 5) (used with -w 4)
             -b [1..100] -> percentage of training set for which to refresh cache
                            when no epsilon violated constraint can be constructed
                            from current cache (default 100%) (used with -w 4)
    SVM-light Options for Solving QP Subproblems (see [3]):
             -n [2..q]   -> number of new variables entering the working set
                            in each svm-light iteration (default n = q).
                            Set n < q to prevent zig-zagging.
             -m [5..]    -> size of svm-light cache for kernel evaluations in MB
                            (default 40) (used only for -w 1 with kernels)
             -h [5..]    -> number of svm-light iterations a variable needs to be
                            optimal before considered for shrinking (default 100)
             -# int      -> terminate svm-light QP subproblem optimization, if no
                            progress after this number of iterations.
                            (default 100000)
    Kernel Options:
             -t int      -> type of kernel function:
                            0: linear (default)
                            1: polynomial (s a*b+c)^d
                            2: radial basis function exp(-gamma ||a-b||^2)
                            3: sigmoid tanh(s a*b + c)
                            4: user defined kernel from kernel.h
             -d int      -> parameter d in polynomial kernel
             -g float    -> parameter gamma in rbf kernel
             -s float    -> parameter s in sigmoid/poly kernel
             -r float    -> parameter c in sigmoid/poly kernel
             -u string   -> parameter of user defined kernel
    Output Options:
             -a string   -> write all alphas to this file after learning
                            (in the same order as in the training set)
    Application-Specific Options:
    
    The following loss functions can be selected with the -l option:
         1  Total number of swapped pairs summed over all queries.
         2  Fraction of swapped pairs averaged over all queries.
    
    References:
        [1] T. Joachims, Training Linear SVMs in Linear Time, Proceedings of the ACM Conference on Knowledge Discovery and Data Mining (KDD), 2006. [Postscript]  [PDF]
        
        [2] T. Joachims, A Support Vector Method for Multivariate Performance Measures, Proceedings of the International Conference on Machine Learning (ICML), 2005. [Postscript]  [PDF]
        
        [3] Tsochantaridis, T. Joachims, T. Hofmann, and Y. Altun, Large Margin Methods for Structured and Interdependent Output Variables, Journal of Machine Learning Research (JMLR), 6(Sep):1453-1484, 2005. [PDF]  
        
        [4] I. Tsochantaridis, T. Hofmann, T. Joachims, Y. Altun. Support Vector Machine Learning for Interdependent and Structured Output Spaces. International Conference on Machine Learning (ICML), 2004. [Postscript]  [PDF]  
        
        [5] T. Joachims, Making Large-Scale SVM Learning Practical. Advances in Kernel Methods - Support Vector Learning, B. Schölkopf and C. Burges and A. Smola (ed.), MIT Press, 1999. [Postscript (gz)] [PDF]
        
        [6] T. Joachims, T. Finley, Chun-Nam Yu, Cutting-Plane Training of Structural SVMs, Machine Learning Journal, 2009. [PDF]
        
        [7] T. Joachims, Optimizing Search Engines Using Clickthrough Data, Proceedings of the ACM Conference on Knowledge Discovery and Data Mining (KDD), ACM, 2002. [Postscript]  [PDF]  

    """
    def __init__(self,arg = None, **kwargs):
        """
        arg: another rankSVM object (deep copy constructor) OR
             path to a saved model OR
             None (default)
        kwargs: Any argument to be passed to the SVM^{rank} implementation
            Any argument in the list of SVM^{rank} arguments can be used (without the dash -)
            For example: rankSVM(t = 2) will create an RBF ranking SVM            
        """
        self.__mstr__ = str(uuid.uuid4())
        self.__trained__ = False
        if type(arg)==type(self):
            self.kwargs = arg.kwargs            
            self.__trained__ = arg.__trained__
            if self.__trained__:
                arg.save(self.__mstr__+'.mdl')
                
        elif type(arg)==type(''):
            self.load(arg)
            self.__trained__ = True
        else:
            self.kwargs = kwargs
            if not len(self.kwargs):
                self.kwargs={'c':0.01}
        
    def fit(self,X,y,g):
        """
        Train the classifier
            X: training examples Nxd
            y: ranks vector (N)
            g: query or group id vector (N)
        """
        assert not self.__trained__
        tmp = self.__mstr__+'.train.dat'
        idx = np.argsort(g)
        dump_svmlight_file(X = X[idx,:],y = y[idx],f = tmp,zero_based = False, query_id = g[idx])
        kw = ''        
        for k in self.kwargs:
            kw += '-'+k+' '+str(self.kwargs[k])+' '
        exe = os.path.join(__svm_rank_path__,'svm_rank_learn')
        cmd = exe+' '+kw+' '+tmp+' '+self.__mstr__ +'.mdl > '+self.__mstr__+'.train.log'        
        t0 = time()        
        if os.system(cmd):
            raise(Exception("Error executing fit (see log) "+ cmd))
        print time()-t0
        os.remove(tmp)
        self.__trained__ = True
    def decision_function(self,X):
        """
        Calculate the decision function given an exmaple X
        """
        assert self.__trained__
        tmp = self.__mstr__+'.test.dat'
        dump_svmlight_file(X = X,y = np.ones(X.shape[0]),f = tmp,zero_based = False)
        out = self.__mstr__+'.test.txt'
        exe = os.path.join(__svm_rank_path__,'svm_rank_classify')
        cmd = exe+' '+tmp+' '+self.__mstr__ +'.mdl '+ out+' > '+self.__mstr__+'.test.log'
        if os.system(cmd):
            raise(Exception("Error executing decision_function (see log) "+ cmd))               
        y = np.loadtxt(out)
        os.remove(tmp) 
        os.remove(out) 
        return y        
        
    def save(self,ofname):
        """
        Save to file (only trained models can be saved)
        """
        assert self.__trained__
        copyfile(self.__mstr__+'.mdl',ofname)
        
    def load(self,ifname):
        """
        Load from file
        """
        copyfile(ifname,self.__mstr__+'.mdl')
        self.__trained__ = True
    def clear(self):
        """
        Remove temp files
        """
        if os.path.exists(self.__mstr__+'.mdl'):
            for f in glob(self.__mstr__+'*'):
                os.remove(f)
            self.__trained__ = False

import matplotlib.pyplot as plt
import itertools

def plotit(X,Y=None,clf=None, markers = ('s','o'), hold = False, transform = None):
    """
    Just a function for showing a data scatter plot and classification boundary
    of a classifier clf (2D), taken from: https://github.com/foxtrotmike/svmtutorial/blob/master/svmtutorial.ipynb
    """
    minx, maxx = np.min(X[:,0]), np.max(X[:,0])
    miny, maxy = np.min(X[:,1]), np.max(X[:,1])
    if clf is not None:
        npts = 100
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        z = clf.decision_function(t)
        z = np.reshape(z,(npts,npts))
        
        extent = [minx,maxx,miny,maxy]
        plt.imshow(z,vmin = -2, vmax = +2)    
        plt.contour(z,[-1,0,1],linewidths = [2],colors=('b','k','r'),extent=extent, label='f(x)=0')
        plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.axis([minx,maxx,miny,maxy])   
    if Y is not None:
        plt.scatter(X[Y==1,0],X[Y==1,1],marker = markers[0], c = 'y', s = 30)
        plt.scatter(X[Y==-1,0],X[Y==-1,1],marker = markers[1],c = 'c', s = 30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')        
         
    else:
        plt.scatter(X[:,0],X[:,1],marker = '.', c = 'k', s = 5)
    if not hold:
        plt.grid()
        plt.show()
            
if __name__=='__main__':
    
    Xp = 1+np.random.randn(50,2)
    Xn = -1-np.random.randn(50,2)
    X = np.vstack((Xp,Xn))
    Y = np.array([1]*Xp.shape[0]+[-1]*Xn.shape[0])
    print 'The data dimensions are',X.shape, Y.shape


    Xpt = 1+np.random.randn(50,2)
    Xnt = -1-np.random.randn(50,2)
    Xt = np.vstack((Xpt,Xnt))
    Yt = np.array([1]*Xpt.shape[0]+[-1]*Xnt.shape[0])    
    
    G = np.random.randint(0,2,len(Y))
    
    rs = SVMrank(c = 10, t = 2, w = 4)
    rs.fit(X,Y,G)
    
    
    yp =  rs.decision_function(Xt)
    from sklearn.metrics import roc_auc_score
    print "ROC",roc_auc_score(Yt,yp)   
    rs.save('my.mdl')
    rs.clear()
    ## just testing a saved model
    rs2 = SVMrank('my.mdl')
    yp =  rs2.decision_function(Xt)
    print "ROC",roc_auc_score(Yt,yp)  
    plotit(X,Y, rs2)
    rs2.clear()