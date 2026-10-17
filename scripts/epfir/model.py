"""独立宽带模型、保守局部校准和配对高频差分；模型不声称还原真实电路。"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares,lsq_linear
from scipy.ndimage import gaussian_filter1d
from .common import ease,smooth
from .analysis import frozen_phase

def ap(f,fc,q):
    x=np.asarray(f)/fc
    return -2*np.arctan2(x/q,1-x*x)

def evaluate(f,p,k,order,timing=True):
    y=np.zeros_like(np.asarray(f),dtype=float)
    for j in range(k):
        y+=ap(f,np.exp(p[2*j]),np.exp(p[2*j+1]))+2*np.pi
    j=2*k
    if order==1:
        y+=2*np.arctan(np.asarray(f)/np.exp(p[j]))-np.pi;j+=1
    elif order in (2,3):
        term=ap(f,np.exp(p[j]),np.exp(p[j+1]))+2*np.pi
        y+=-term if order==2 else term;j+=2
    if timing:
        y+=2*np.pi*np.asarray(f)*p[-1]*1e-6
    return y

def fit(a,c):
    f=a['f'];source=a['phase'];conf=a['confidence']
    mask=(f>=a['min_usable_hz'])&(f<=8000)&(conf>.12)
    indices=np.flatnonzero(mask)[::3]
    if len(indices)<30:
        raise ValueError(f"{a['name']} 可用宽带频点不足，需重新测量或确认直达峰")
    # 宽带模型用未平滑相位的圆周残差，避免窄带完整旋转经平滑后变成虚假的宽带失配。
    # 连续平滑相位仍保留，用于局部群延时和固定规约的前后比较。
    ff=f[indices];yy=a['phase_raw'][indices];ww=np.sqrt(conf[indices])
    # 低频房间旋转不决定分频模型；保留其局部诊断。
    ww*=.25+.75*ease((ff-200)/500)
    known=a['crossover_hz'];known=[known] if np.isscalar(known) else list(known)
    ignored=[x for x in known if not 8<=float(x)<8000]
    if ignored:
        a['notes'].append('位于可拟合频带外的分频点不作硬约束。')
    known=[float(x) for x in known if 8<=float(x)<8000]
    counts=sorted(set([max(0,len(known)),max(1,len(known))])) if known else [0,1,2,3]
    if c.get('existing_phase_correction')=='this_filter':
        # 分频先验描述物理系统；已校准后的残差不必再含一份分频旋转。
        counts=sorted(set([0]+counts))
    records=[]
    for k in counts:
        seeds=known if len(known)==k else np.geomspace(650,4200,k).tolist()
        if k==1 and not known:
            seeds=[2100.]
        for order in (0,1,2,3):
            p=[];lo=[];hi=[]
            for fc in seeds:
                p.extend([np.log(fc),np.log(.707)])
                lo.extend([np.log(max(8,fc/3)),np.log(.3)])
                hi.extend([np.log(min(8000,fc*3)),np.log(2.)])
            if order:
                p.append(np.log(50));lo.append(np.log(8));hi.append(np.log(180))
                if order in (2,3):
                    p.append(np.log(.7));lo.append(np.log(.3));hi.append(np.log(2.))
            p.append(0.);lo.append(-100.);hi.append(100.)
            def residual(v):
                e=evaluate(ff,v,k,order)-yy
                return np.r_[np.sin(e)*ww,(1-np.cos(e))*ww]
            result=least_squares(residual,p,bounds=(lo,hi),loss='soft_l1',f_scale=.12,
                                 x_scale='jac',max_nfev=250)
            err=np.angle(np.exp(1j*(evaluate(ff,result.x,k,order)-yy)))
            # 留一频段式评分：每四点的第四点不进入下方重新拟合。
            keep=np.arange(len(ff))%4!=3
            def train(v):
                z=residual(v).reshape(2,-1)
                return z[:,keep].ravel()
            held=least_squares(train,result.x,bounds=(lo,hi),loss='soft_l1',f_scale=.12,max_nfev=60)
            ee=np.angle(np.exp(1j*(evaluate(ff,held.x,k,order)-yy)))
            score=float(np.average(np.minimum(ee[~keep]**2,.6**2),weights=ww[~keep]**2))+.0018*len(p)
            records.append(dict(k=k,order=order,parameters=result.x,score=score,
                                rms_deg=float(np.rad2deg(np.sqrt(np.average(err**2,weights=ww**2))))))
    best=min(records,key=lambda z:z['score'])
    p=best['parameters'];base=evaluate(f,p,best['k'],best['order'],False)
    nuisance=2*np.pi*f*p[-1]*1e-6
    normalized=frozen_phase(f,a['phase_raw']-nuisance,a['smoothing_input_valid'],a['smoothing_selection'],c)[0]
    # 未知/非标准分频可用平滑曲线，整数圈先按宽带模型就近对齐。
    branch=base+np.angle(np.exp(1j*(a['phase_raw']-nuisance-base)))
    fallback=best['rms_deg']>24
    if fallback:
        broad=smooth(f,branch,1.)
        blend=ease((f-300)/300)*(1-ease((f-7000)/1000))
        base=base*(1-blend)+broad*blend
    residual=np.angle(np.exp(1j*(normalized-base)))
    residual=smooth(f,residual,c['smoothing_octaves'])
    # 由延迟预算确定可表示的最窄频率结构，另有 FDW 时间约束。
    spacing=float(np.median(np.diff(np.log2(f))))
    delay=c['delay_samples']/c['sample_rate']
    if delay>0:
        hz=np.linspace(0,c['lf_stop_hz']*1.5,2401)
        rr=np.interp(hz,f,residual)
        rr=gaussian_filter1d(rr,max(1,(.65/delay)/(hz[1]-hz[0])/2.35482))
        residual=np.interp(f,hz,rr)
    else:
        residual*=0
    head=c['head_half_ranges_cm']
    radius=np.linalg.norm(head)/100 if head is not None else .2
    move=343/(8*radius) if radius>0 else float('inf')
    spatial=1/(1+(f/move)**4)
    low=a['min_usable_hz']
    local_guard=ease((f-low)/max(8,low*.35))*(1-ease((f-c['lf_local_high_hz'])/(c['lf_stop_hz']-c['lf_local_high_hz'])))
    # 相位角在整数圈边界有歧义，局部项以真实连续相位斜率为主拟合。
    # 频率基函数宽度同时受 FIR 前时间限制，不能靠密集频点伪造自由度。
    from scipy.interpolate import PchipInterpolator
    grid=np.arange(low,c['lf_local_high_hz']+1,1.)
    sigma=max(9.,.32/max(delay,.001))
    centers=np.arange(low,c['lf_local_high_hz']+1,max(10.,sigma))
    def basis(x):
        guard=ease((x-low)/max(8,low*.35))*(1-ease((x-c['lf_local_high_hz'])/(c['lf_stop_hz']-c['lf_local_high_hz'])))
        return np.exp(-.5*((x[:,None]-centers)/sigma)**2)*guard[:,None]
    B=basis(grid);Bd=(basis(grid+.02)-basis(grid-.02))/.04
    slope=PchipInterpolator(f,normalized-base).derivative()(grid)
    weights=np.sqrt(np.interp(grid,f,conf*spatial))
    angle=np.angle(np.exp(1j*np.interp(grid,f,normalized-base)))
    regular=np.diff(np.eye(len(centers)),n=2,axis=0)
    edge=np.linspace(c['lf_local_high_hz'],c['lf_stop_hz'],25)
    E=basis(edge);Ed=(basis(edge+.02)-basis(edge-.02))/.04
    # 退出区的相位与斜率都要接回宽带基线，不能把核心区的相位量挤到这里归零。
    matrix=np.vstack([B*weights[:,None]*.15,Bd*weights[:,None]*50,E*3,Ed*75,regular*.4,np.eye(len(centers))*.15])
    def solve_local(reference):
        slope=PchipInterpolator(f,normalized-reference).derivative()(grid)
        angle=np.angle(np.exp(1j*np.interp(grid,f,normalized-reference)))
        rhs=np.r_[-angle*weights*.15,-slope*weights*50,np.zeros(2*len(edge)+len(regular)+len(centers))]
        coeff=lsq_linear(matrix,rhs,bounds=(-np.deg2rad(60),np.deg2rad(60))).x
        return basis(f)@coeff
    local=solve_local(base)
    lf_model=evaluate(f,p,0,best['order'],False) if best['k']==0 else evaluate(f,p,best['k'],best['order'],False)-evaluate(f,p,best['k'],0,False)
    without_lf=base-lf_model
    alternative_local=solve_local(without_lf)
    end=min(a.get('max_usable_hz',c['independent_end_hz']),c['independent_end_hz'])
    start=min(c['independent_transition_hz'],end*.8)
    if end<c['lf_stop_hz']:
        local*=1-ease((f-end*.8)/(end*.2))
        alternative_local*=1-ease((f-end*.8)/(end*.2))
    independent=-base*(1-ease((f-start)/(end-start)))
    a.update(model_phase=base,normalized_phase=normalized,nuisance_phase=nuisance,
             local_target=local,independent_target=independent,pair_target=np.zeros(len(f)),
             without_lf_target=-without_lf*(1-ease((f-start)/(end-start))),local_without_lf_target=alternative_local,
             model=dict(**best,fallback=fallback,candidates=records,
                        nuisance_advance_us=float(p[-1]),timing_applied_to_audio=False,
                        movement_prior_hz=float(move),spatial_evidence=False))
    return a

def pair_models(analyses,c):
    lookup={a['name']:a for a in analyses}
    for left,right in c.get('pairs',[]):
        a,b=lookup[left],lookup[right];f=a['f']
        phi_b=np.interp(f,b['f'],b['normalized_phase'])
        delta=np.unwrap(np.angle(np.exp(1j*(a['normalized_phase']-phi_b))))
        at=(f>=8000)&(f<=10000)
        delta-=2*np.pi*np.rint(np.median(delta[at])/(2*np.pi))
        delta=smooth(f,delta,1/3)
        # 对称的小幅差分校准，不拟合各侧高频细节或把共同剩余相位压到零。
        reliability=np.minimum(a['confidence'],np.interp(f,b['f'],b['confidence']))
        reliability=np.clip(reliability/.5,0,1)
        weight=ease((f-8000)/2000)*(1-ease((f-10000)/(c['pair_end_hz']-10000)))
        target=-.5*np.clip(delta,-np.pi,np.pi)*weight*reliability
        a['pair_target']=target
        b['pair_target']=np.interp(b['f'],f,-target)
        a['pair_with']=right;b['pair_with']=left
    return analyses
