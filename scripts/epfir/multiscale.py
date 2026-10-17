"""统一频带上的正则化慢变相位基底，以及有独立支持范围的局部细节项。"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import PchipInterpolator
from .common import ease,smooth

def ranges(a,c):
    head=c['head_half_ranges_cm']
    radius=float(np.linalg.norm(head)/100) if head is not None else None
    estimated=343/(8*radius) if radius is not None and radius>0 else (float('inf') if radius==0 else 200.)
    cap=float(c.get('local_detail_cap_hz',300.))
    upper=min(cap,estimated,a.get('max_usable_hz',c['independent_end_hz']))
    low=a['min_usable_hz'];end=min(a.get('max_usable_hz',10000.),c['independent_end_hz'])
    return dict(global_low_hz=low,global_high_hz=end,global_transition_start_hz=min(8000.,end*.8),
                local_limit_hz=upper,local_fade_start_hz=max(low,upper*.8),
                movement_estimate_hz=estimated,head_radius_m=radius,
                local_cap_hz=cap,spatial_validation=False,
                unknown_movement=head is None,movement_affects='local_detail_only')

def trend(a,reference,c):
    """在整段可信频带拟合慢变残差；头部范围不削弱这个基底。"""
    f=a['f'];r=a['normalized_phase']-reference;band=ranges(a,c)
    low=band['global_low_hz'];high=band['global_high_hz'];start=band['global_transition_start_hz']
    centers=np.arange(np.log2(low*.7),np.log2(high)+.25,.25)
    width=float(c.get('global_basis_sigma_octaves',.65))
    # 不使用带外测量作反演，只把拟合出的边界平顺延伸至 DC，避免硬起点的长尾。
    guard=ease(f/low)
    B=np.exp(-.5*((np.log2(f)[:,None]-centers)/width)**2)*guard[:,None]
    valid=(f>=low)&(f<=start)&a['valid']&(a['confidence']>.12)
    if valid.sum()<25:
        raise ValueError('整段平滑相位的可信频点不足')
    rows=B[valid];target=r[valid]
    # 只允许整体整数圈重标；不在某个局部频段随意改变圈数。
    target=target-2*np.pi*np.rint(np.median(target[-max(5,len(target)//8):])/(2*np.pi))
    original=np.sqrt(a['confidence'][valid]);weight=original.copy()
    D=np.diff(np.eye(len(centers)),n=2,axis=0)
    lam=float(c.get('global_regularization',3.))
    coeff=np.zeros(len(centers))
    for _ in range(3):
        matrix=np.vstack([rows*weight[:,None],lam*D,.03*lam*np.eye(len(centers))])
        rhs=np.r_[target*weight,np.zeros(len(D)+len(centers))]
        coeff=lsq_linear(matrix,rhs,bounds=(-np.pi,np.pi),tol=1e-9).x
        error=rows@coeff-target
        weight=original*np.sqrt(np.minimum(1,np.deg2rad(7)/np.maximum(abs(error),1e-12)))
    result=B@coeff
    metadata=dict(basis='log_frequency_gaussian_rbf',spacing_octaves=.25,sigma_octaves=width,
                  roughness_penalty=lam,robust_iterations=3,coefficients=coeff,
                  centers_hz=2**centers,head_weight_applied=False,
                  residual_rms_deg=float(np.rad2deg(np.sqrt(np.mean((rows@coeff-target)**2)))))
    return result,metadata

def local(a,reference,c):
    """只拟合慢变基底之外的快速残差；相位和斜率在局部上限前归零。"""
    f=a['f'];band=ranges(a,c);low=band['global_low_hz'];upper=band['local_limit_hz']
    if upper<=low+15 or c['delay_samples']==0:
        return np.zeros(len(f))
    residual=a['normalized_phase']-reference
    # 慢变残差归全局基底，禁止局部项携带宽带相位台阶。
    fast=residual-smooth(f,residual,c.get('local_highpass_octaves',.8))
    grid=np.arange(low,upper+1,1.);delay=c['delay_samples']/c['sample_rate']
    sigma=max(9.,.32/max(delay,.001));centers=np.arange(low,upper+1,max(10.,sigma))
    start=band['local_fade_start_hz']
    def basis(x):
        guard=ease((x-low)/max(8,low*.35))*(1-ease((x-start)/max(upper-start,1)))
        return np.exp(-.5*((x[:,None]-centers)/sigma)**2)*guard[:,None]
    B=basis(grid);Bd=(basis(grid+.02)-basis(grid-.02))/.04
    estimated=band['movement_estimate_hz']
    spatial=1/(1+(grid/estimated)**4) if np.isfinite(estimated) else np.ones(len(grid))
    weights=np.sqrt(np.interp(grid,f,a['confidence'])*spatial)
    derivative=PchipInterpolator(f,fast).derivative()(grid)
    D=np.diff(np.eye(len(centers)),n=2,axis=0)
    edge=np.linspace(start,upper,18);E=basis(edge);Ed=(basis(edge+.02)-basis(edge-.02))/.04
    # 压制局部项的平均量，避免它夺走本应由全局基底表达的缓慢偏移。
    moment=(weights[:,None]*B).mean(axis=0,keepdims=True)
    A=np.vstack([B*weights[:,None]*.25,Bd*weights[:,None]*50,.6*D,.15*np.eye(len(centers)),E*1.5,Ed*50,moment*5])
    y=np.r_[-np.interp(grid,f,fast)*weights*.25,-derivative*weights*50,
            np.zeros(len(D)+len(centers)+2*len(edge)+1)]
    coeff=lsq_linear(A,y,bounds=(-np.deg2rad(60),np.deg2rad(60))).x
    return basis(f)@coeff

def prepare(a,c):
    f=a['f'];band=ranges(a,c);end=band['global_high_hz'];start=band['global_transition_start_hz']
    guard=1-ease((f-start)/(end-start))
    references=[('full_model',a['model_phase'])]
    if a['model']['order']:
        references.append(('without_uncertain_lf_model',-a['without_lf_target']/np.maximum(guard,1e-12)))
    variants=[]
    for name,reference in references:
        delta,metadata=trend(a,reference,c)
        variants.append(dict(name=name,reference=reference,parametric_target=-reference,
                             smooth_target=-delta,smooth_forward=delta,regularization=metadata))
    a['global_variants']=variants;a['calibration_ranges']=band
    return a
