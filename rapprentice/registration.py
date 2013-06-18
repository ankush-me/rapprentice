"""
Register point clouds to each other
"""

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
from rapprentice import tps, svds, math_utils
# from svds import svds


"""
arrays are named like name_abc
abc are subscripts and indicate the what that tensor index refers to

index name conventions:
    m: test point index
    n: training point index
    a: input coordinate
    g: output coordinate
    d: gripper coordinate
"""

class Transformation(object):
    """
    Object oriented interface for transformations R^d -> R^d
    """
    def transform_points(self, x_ma):
        raise NotImplementedError
    def compute_jacobian(self, x_ma):
        raise NotImplementedError        

        
    def transform_bases(self, x_ma, rot_mad, orthogonalize=True, orth_method = "cross"):
        """
        orthogonalize: none, svd, qr
        """

        grad_mga = self.compute_jacobian(x_ma)
        newrot_mgd = np.array([grad_ga.dot(rot_ad) for (grad_ga, rot_ad) in zip(grad_mga, rot_mad)])
        

        if orthogonalize:
            if orth_method == "qr": 
                newrot_mgd =  orthogonalize3_qr(newrot_mgd)
            elif orth_method == "svd":
                newrot_mgd = orthogonalize3_svd(newrot_mgd)
            elif orth_method == "cross":
                newrot_mgd = orthogonalize3_cross(newrot_mgd)
            else: raise Exception("unknown orthogonalization method %s"%orthogonalize)
        return newrot_mgd
        
    def transform_hmats(self, hmat_mAD):
        """
        Transform (D+1) x (D+1) homogenius matrices
        """
        hmat_mGD = np.empty_like(hmat_mAD)
        hmat_mGD[:,:3,3] = self.transform_points(hmat_mAD[:,:3,3])
        hmat_mGD[:,:3,:3] = self.transform_bases(hmat_mAD[:,:3,3], hmat_mAD[:,:3,:3])
        return hmat_mGD
        
    def compute_numerical_jacobian(self, x_d, epsilon=0.0001):
        "numerical jacobian"
        x0 = np.asfarray(x_d)
        f0 = self.transform_points(x0)
        jac = np.zeros(len(x0), len(f0))
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (self.transform_points(x0+dx) - f0) / epsilon
            dx[i] = 0.
        return jac.transpose()

class ThinPlateSpline(Transformation):
    """
    members:
        x_na: centers of basis functions
        w_ng: 
        lin_ag: transpose of linear part, so you take x_na.dot(lin_ag)
        trans_g: translation part
    
    """
    def __init__(self, d=3):
        "initialize as identity"
        self.x_na = np.zeros((0,d))
        self.lin_ag = np.eye(d)
        self.trans_g = np.zeros(d)
        self.w_ng = np.zeros((0,d))

    def transform_points(self, x_ma):
        y_ng = tps.tps_eval(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return y_ng
    def compute_jacobian(self, x_ma):
        grad_mga = tps.tps_grad(x_ma, self.lin_ag, self.trans_g, self.w_ng, self.x_na)
        return grad_mga
        

def fit_ThinPlateSpline(x_na, y_ng, bend_coef=.1, rot_coef = 1e-5, wt_n=None):
    """
    x_na: source cloud
    y_nd: target cloud
    smoothing: penalize non-affine part
    angular_spring: penalize rotation
    wt_n: weight the points        
    """
    f = ThinPlateSpline()
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit3(x_na, y_ng, bend_coef, rot_coef, wt_n)
    f.x_na = x_na
    return f        


def fit_ThinPlateSpline_RotReg(x_na, y_ng, bend_coef = .1, wt_n=None, rot_coefs = (0.01,0.01,0.0025),scale_coef=.01, niter_powell=100):
    import fastrapp
    f = ThinPlateSpline()
    rfunc = fastrapp.rot_reg
    fastrapp.set_coeffs(rot_coefs, scale_coef)
    f.lin_ag, f.trans_g, f.w_ng = tps.tps_fit_regrot(x_na, y_ng, bend_coef, rfunc, wt_n, max_iter=niter_powell)
    f.x_na = x_na
    return f        

def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))    

def loglinspace_arr(a,b,n):
    "n vectors b/w a and b with constant ratio b/w them"
    assert len(a)==len(b)
    return np.r_[[loglinspace(a[i], b[i], n) for i in xrange(len(a))]].T


def tps_rpm(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .05, rad_final = .001, 
            plotting = False, verbose=True, f_init = None, return_full = False, plot_cb = None):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    if f_init is not None: 
        f = f_init  
    else:
        f = ThinPlateSpline(d)
        f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        # targ_nd = find_targets(x_nd, y_md, corr_opts = dict(r = rads[i], p = .1))
        corr_nm = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.2)

        wt_n = corr_nm.sum(axis=1)

        goodn = wt_n > .1

        targ_Nd = np.dot(corr_nm[goodn, :]/wt_n[goodn][:,None], y_md)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f)
        
        
        x_Nd = x_nd[goodn]
        f = fit_ThinPlateSpline(x_Nd, targ_Nd, bend_coef = regs[i], wt_n = wt_n[goodn], rot_coef = 10*regs[i])


    if return_full:
        info = {}
        info["corr_nm"] = corr_nm
        info["goodn"] = goodn
        info["x_Nd"] = x_nd[goodn,:]
        info["targ_Nd"] = targ_Nd
        info["wt_N"] = wt_n[goodn]
        info["cost"] = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, x_Nd, targ_Nd, regs[-1])
        return f, info
    else:
        return f



def tps_rpm_regrot(x_nd, y_md, n_iter = 100, bend_init = 0.05, bend_final = .0001, 
                   rot_init = (0.1,0.1,0.025), rot_final=(0.001,0.001,0.00025), scale_init=1, scale_final=0.001, rad_init = .5, rad_final = .0005,
                   verbose=True, f_init = None, return_full = False):
    """
    tps-rpm which uses regularization on the rotation and scaling separately.
    Based on John's [Chui & Rangarajan] tps-rpm.
    
    Various parameters are:
    ======================================
    
        1. bend_init/final : Regularization on the non-linear part of the transform.
                             This value should go to zero to get exact fit.
        
        2. rot_init/final (x,y,z) : Regularization on the rotation along the x,y,z axes.
                                    Note that the default cost for rotation along z is much lower than
                                    the cost for rotation along the x and y axes.
        
        3. scale_init/final : Regularization on the scaling part of the affine transform.
                              This should in general be high because normally the scene is not scaled.
        
        4. rad_init/final   : This defines the unit of distance in each iteration.
                              The correspondences are calculated as exp(- dist/ r).
                              This parameter defines this r. 
                              [see Chui and Rangarajan page 5, equation 6. This r is the 'T' there.]

    Possible scope for improvement/ exploration:
        Currently, in every iteration, all the parameters are updated, 
        may be we need to add more loops, iterating over the parameters individually.
    """
    _,d  = x_nd.shape
    regs = loglinspace(bend_init, bend_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    scales = loglinspace(scale_init, scale_final, n_iter)
    rots = loglinspace_arr(rot_init, rot_final, n_iter)
    

    # initialize the function f.
    if f_init is not None: 
        f = f_init  
    else:
        f         = ThinPlateSpline(d)
        f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
        

    # iterate b/w calculating correspondences and fitting the transformation.
    for i in xrange(n_iter):
        print 'iter : ', i
        xwarped_nd = f.transform_points(x_nd)
        corr_nm = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.2)

        wt_n = corr_nm.sum(axis=1)
        goodn = wt_n > .1
        targ_Nd = np.dot(corr_nm[goodn, :]/wt_n[goodn][:,None], y_md)

        x_Nd = x_nd[goodn]
        f = fit_ThinPlateSpline_RotReg(x_Nd, targ_Nd, wt_n=wt_n[goodn], bend_coef = regs[i], rot_coefs = rots[i], scale_coef=scales[i])


    if return_full:
        info = {}
        info["corr_nm"] = corr_nm
        info["goodn"] = goodn
        info["x_Nd"] = x_nd[goodn,:]
        info["targ_Nd"] = targ_Nd
        info["wt_N"] = wt_n[goodn]
        info["cost"] = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, x_Nd, targ_Nd, regs[-1])
        return f, info
    else:
        return f


def tps_rpm_regrot_multi(x_clouds, y_clouds, n_iter = 100, 
                   n_iter_powell_init=100, n_iter_powell_final=100, 
                   bend_init = 0.05, bend_final = .0001, 
                   rot_init = (0.1,0.1,0.025), rot_final=(0.001,0.001,0.00025),
                   scale_init=1, scale_final=0.001, 
                   rad_init = .5, rad_final = .0005,
                   verbose=True, f_init = None, return_full = False,
                   plotting_cb=None):
    """
    Similar to tps_rpm_regrot except that it accepts a 
    LIST of source and target point clouds and registers 
    a cloud in the source to the corresponding one in the target.  
    
    For details on the various parameters check the doc of tps_rpm_regrot.
    """

    assert len(x_clouds)==len(y_clouds), "Different number of point-clouds in source and target."
    
    #flatten the list of point clouds into one big point cloud
    combined_x = np.concatenate(x_clouds) 
    combined_y = np.concatenate(y_clouds)

    # concatenate the clouds into one big cloud
    _,d  = combined_x.shape

    regs     = loglinspace(bend_init, bend_final, n_iter)
    rads     = loglinspace(rad_init, rad_final, n_iter)
    scales   = loglinspace(scale_init, scale_final, n_iter)
    rots     = loglinspace_arr(rot_init, rot_final, n_iter)
    npowells = loglinspace(n_iter_powell_init, n_iter_powell_final, n_iter).astype(int) 

    # initialize the function f.
    if f_init is not None: 
        f = f_init  
    else:
        f         = ThinPlateSpline(d)
        f.trans_g = np.median(combined_y,axis=0) - np.median(combined_x,axis=0)


    # iterate b/w calculating correspondences and fitting the transformation.
    for i in xrange(n_iter):
        print 'iter : ', i        
        target_pts   = []
        good_inds    = []
        wt           = []

        for j in xrange(len(x_clouds)): #process a pair of point-clouds
            x_nd = x_clouds[j]
            y_md = y_clouds[j]
            
            assert x_nd.ndim==y_md.ndim==2, "tps_rpm_reg_rot_multi : Point clouds are not two dimensional arrays"
                        
                        
            xwarped_nd = f.transform_points(x_nd)
            corr_nm = calc_correspondence_matrix(xwarped_nd, y_md, r=rads[i], p=.2)

            wt_n = corr_nm.sum(axis=1) # gives the row-wise sum of the corr_nm matrix
            goodn = wt_n > .1
            targ_Nd = np.dot(corr_nm[goodn, :]/wt_n[goodn][:,None], y_md) # calculate the average points based on softmatching

            target_pts.append(targ_Nd)
            good_inds.append(goodn)  
            wt.append(wt_n[goodn])

        target_pts = np.concatenate(target_pts)
        good_inds  = np.concatenate(good_inds)
        source_pts = combined_x[good_inds]
        wt         = np.concatenate(wt)

        assert len(target_pts)==len(source_pts)==len(wt), "Lengths are not equal. Error!"
        f = fit_ThinPlateSpline_RotReg(source_pts, target_pts, wt_n=wt, bend_coef = regs[i], rot_coefs = rots[i], scale_coef=scales[i], niter_powell=npowells[i])
        
        from rapprentice.colorize import colorize
        print colorize("\ttps-rpm-rotreg : iter : %d | fit distance : "%i, "red", True) , colorize("%g"%match_score(f.transform_points(source_pts), target_pts), "green", True)

        if plotting_cb and i%5==0:
            plotting_cb(f)
        

    if return_full:
        info = {}
        info["cost"] = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, source_pts, target_pts, regs[-1])
        return f, info
    else:
        return f


def match_score(src, target):
    """
    lower score ==> better match
    """
    d = np.sum(np.abs(src-target)**2,axis=-1)**(1./2)
    return d.sum()


def logmap(m):
    "http://en.wikipedia.org/wiki/Axis_angle#Log_map_from_SO.283.29_to_so.283.29"
    theta = np.arccos(np.clip((np.trace(m) - 1)/2,-1,1))
    return (1/(2*np.sin(theta))) * np.array([[m[2,1] - m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]]]), theta
# 
# def tps_rpm_zrot(x_nd, y_md, n_iter = 5, reg_init = .1, reg_final = .001, rad_init = .2, rad_final = .001, plotting = False, 
#                  verbose=True, rot_param=(.05, .05, .05), scale_param = .01, plot_cb = None):
#     """
#     Do tps_rpm algorithm for each z angle rotation
#     Then don't reestimate affine part in tps optimization
#     
#     rot param: meters error per point per radian
#     scale param: meters error per log2 scaling
#     
#     """
#     
#     n_initializations = 5
#     
#     n,d = x_nd.shape
#     regs = loglinspace(reg_init, reg_final, n_iter)
#     rads = loglinspace(rad_init, rad_final, n_iter)
#     zrots = np.linspace(-np.pi/2, pi/2, n_initializations)
# 
#     costs,tpscosts,regcosts = [],[],[]
#     fs = []
#     
#     # convert in to the right units: meters/pt -> meters*2
#     rot_coefs = np.array(rot_param)
#     scale_coef = scale_param
#     import fastmath
#     fastmath.set_coeffs(rot_coefs, scale_coef)
#     #def regfunc(b):        
#         #if np.linalg.det(b) < 0 or np.isnan(b).any(): return np.inf
#         #b = b.T
#         #u,s,vh = np.linalg.svd(b)
#         ##p = vh.T.dot(s.dot(vh))        
#         #return np.abs(np.log(s)).sum()*scale_coef + float(np.abs(logmap(u.dot(vh))).dot(rot_coefs))
#     regfunc = fastmath.rot_reg
#     reggrad = fastmath.rot_reg_grad
#     
#     for a in zrots:
#         f_init = ThinPlateSplineRegularizedLinearPart(regfunc, reggrad=reggrad)
#         f_init.n_fit_iters = 2
#         f_init.lin_ag[:2,:2] = np.array([[cos(a), sin(a)],[-sin(a), cos(a)]])
#         f_init.trans_g =  y_md.mean(axis=0) - f_init.transform_points(x_nd).mean(axis=0)
#         f, info = tps_rpm(x_nd, y_md, n_iter=n_iter, reg_init=reg_init, reg_final=reg_final, rad_init = rad_init, rad_final = rad_final, plotting=plotting, verbose=verbose, f_init=f_init, return_full=True, plot_cb=plot_cb)
#         ypred_ng = f.transform_points(x_nd)
#         dists_nm = ssd.cdist(ypred_ng, y_md)
#         # how many radians rotation is one mm average error reduction worth?
# fit_score(src, targ, dist_param):
#         tpscost = info["cost"]
#         # seems like a reasonable goodness-of-fit measure
#         regcost = regfunc(f.lin_ag)
#         tpscosts.append(dists_nm.min(axis=1).mean())
#         regcosts.append(regcost)
#         costs.append(tpscost + regcost)
#         fs.append(f)        
#         print "linear part", f.lin_ag
#         u,s,vh = np.linalg.svd(f.lin_ag)
#         print "angle-axis:",logmap(u.dot(vh))
#         print "scaling:", s
#         
#     print "zrot | tps | reg | total"
#     for i in xrange(len(zrots)):
#         print "%.5f | %.5f | %.5f | %.5f"%(zrots[i], tpscosts[i], regcosts[i], costs[i])
# 
#     i_best = np.argmin(costs)
#     best_f = fs[i_best]
#     print "best initialization angle", zrots[i_best]*180/np.pi
#     u,s,vh = np.linalg.svd(best_f.lin_ag)
#     print "best rotation axis,angle:",logmap(u.dot(vh))
#     print "best scaling:", s
# 
#     if plotting:
#         plot_cb(x_nd, y_md, None, None, None, f)
# 
#     return best_f


def find_targets(x_md, y_nd, corr_opts):
    """finds correspondence matrix, and then for each point in source cloud,
    find the weighted average of its "partners" in the target cloud"""

    corr_mn = calc_correspondence_matrix(x_md, y_nd, **corr_opts)
    # corr_mn = M.match(x_md, y_nd)
    # corr_mn = corr_mn / corr_mn.sum(axis=1)[:,None]
    return np.dot(corr_mn, y_nd)        

def calc_correspondence_matrix(x_nd, y_md, r, p, n_iter=20):
    """
    sinkhorn procedure. see tps-rpm paper
    TODO ask jonathan about geometric mean hack
    """
    n = x_nd.shape[0]
    m = y_md.shape[0]
    dist_nm = ssd.cdist(x_nd, y_md,'euclidean')
    prob_nm = np.exp(-dist_nm / r)
    prob_nm_orig = prob_nm.copy()
    for _ in xrange(n_iter):
        prob_nm /= (p*((n+0.)/m) + prob_nm.sum(axis=0))[None,:]  # cols sum to n/m
        prob_nm /= (p + prob_nm.sum(axis=1))[:,None] # rows sum to 1

    prob_nm = np.sqrt(prob_nm_orig * prob_nm)
    prob_nm /= (p + prob_nm.sum(axis=1))[:,None] # rows sum to 1
    return prob_nm

def nan2zero(x):
    np.putmask(x, np.isnan(x), 0)
    return x


def fit_score(src, targ, dist_param):
    "how good of a partial match is src to targ"
    sqdists = ssd.cdist(src, targ,'sqeuclidean')
    return -np.exp(-sqdists/dist_param**2).sum()

def orthogonalize3_cross(mats_n33):
    "turns each matrix into a rotation"

    x_n3 = mats_n33[:,:,0]
    y_n3 = mats_n33[:,:,1]
    # z_n3 = mats_n33[:,:,2]

    xnew_n3 = math_utils.normr(x_n3)
    znew_n3 = math_utils.normr(np.cross(xnew_n3, y_n3))
    ynew_n3 = math_utils.normr(np.cross(znew_n3, xnew_n3))

    return np.concatenate([xnew_n3[:,:,None], ynew_n3[:,:,None], znew_n3[:,:,None]],2)

def orthogonalize3_svd(x_k33):
    u_k33, _s_k3, v_k33 = svds.svds(x_k33)
    return (u_k33[:,:,:,None] * v_k33[:,None,:,:]).sum(axis=2)

def orthogonalize3_qr(_x_k33):
    raise NotImplementedError