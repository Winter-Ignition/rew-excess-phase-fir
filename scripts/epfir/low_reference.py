"""噪声下限之外拟合低端幅度延伸；模型不是箱体参数识别。"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from .common import db, ease, smooth, minimum


def curve(f, p, order):
    gain, logfc, tilt = p
    x = np.maximum(np.asarray(f, float), 1e-8)
    # 等效 Butterworth 高通幅度，附平滑的有限带宽房间倾斜。
    return gain-10*np.logaddexp(0, 2*order*(logfc-np.log(x)))/np.log(10)+tilt*np.log2(x/100)


def reference(fn, magnitude, cutoff, frozen=None):
    """返回延伸幅度与可冻结模型。低于 cutoff 的观测从不进入拟合。"""
    if cutoff is None:
        return magnitude.copy(), dict(policy='unmodelled_full_band', models=[])
    cutoff=float(cutoff)
    if frozen is None:
        lo=cutoff; hi=min(200., max(100., cutoff*5), fn[-1]*.7)
        if hi<=lo*1.6:
            raise ValueError('有效低频范围不足，无法拟合低端延伸')
        x=np.geomspace(lo,hi,220)
        keep=fn>=lo
        y=smooth(x,np.interp(x,fn[keep],db(magnitude[keep])),1/6)
        weight=np.sqrt(ease((x-lo)/max(2.,lo*.2)))
        records=[]
        for order in (2,3,4,5,6,8,10,12):
            result=least_squares(lambda p:(curve(x,p,order)-y)*weight,
                [float(np.median(y)), np.log(max(lo,35.)),0.],
                bounds=([-250,np.log(2),-8],[250,np.log(250),8]),loss='soft_l1',f_scale=1.)
            err=curve(x,result.x,order)-y
            score=float(np.sqrt(np.average(err**2,weights=weight**2)))
            records.append(dict(order=order,parameters=result.x.tolist(),rms_db=score))
        records.sort(key=lambda r:r['rms_db'])
        models=[r for r in records if r['rms_db']<=records[0]['rms_db']+.35][:6]
        frozen=dict(policy='equivalent_highpass_ensemble', cutoff_hz=cutoff, fit_high_hz=hi,
            models=models, identification='equivalent_rolloff_not_physical_box_order')
    else:
        if abs(frozen['cutoff_hz']-cutoff)>1e-8:
            raise ValueError('复测有效测量下限与冻结参考不一致；请单独做敏感度分析')
        # 冻结形状；只消除本次导出相对电平差，不重新拟合 fc、阶数或倾斜。
        baseline=frozen['models'][0]
        align=(fn>=max(cutoff*2,60.))&(fn<=frozen['fit_high_hz'])
        offset=float(np.median(db(magnitude[align])-curve(fn[align],baseline['parameters'],baseline['order']))) if align.any() else 0.
        adjusted=[]
        for candidate in frozen['models']:
            params=list(candidate['parameters']);params[0]+=offset
            adjusted.append({**candidate,'parameters':params})
        frozen={**frozen, 'models':adjusted,'reused':True,'level_offset_db':offset}
    best=frozen['models'][0]
    extrap=curve(fn,best['parameters'],best['order'])
    join=cutoff*1.35
    blend=ease((fn-cutoff)/(join-cutoff))
    extended=10**(((1-blend)*extrap+blend*db(magnitude))/20)
    # DC 的真实高通零点不作为测得的噪声幅度参与倒谱。
    return np.maximum(extended,1e-15), frozen


def model_spread(fn, magnitude, model, nfft):
    """相同观测和 FFT 下，仅改变候选延伸的最小相位，反映参考不确定度。"""
    if not model.get('models'):
        return np.zeros(len(fn))
    phases=[]
    for candidate in model['models']:
        m={**model,'models':[candidate]}
        ext,_=reference(fn,magnitude,model['cutoff_hz'],m)
        phases.append(minimum(ext,nfft))
    # 圆周距离避免把整数圈当成模型不确定度。
    return np.max(abs(np.angle(np.asarray(phases)/phases[0])),axis=0)
