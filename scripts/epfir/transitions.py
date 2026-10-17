"""按实际校准范围放大频段，同时保留完整峰值。"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def build(analyses,c,out):
    fig,axes=plt.subplots(4,len(analyses),figsize=(7*len(analyses),12),squeeze=False,layout='constrained')
    for j,a in enumerate(analyses):
        f=a['f'];before=a.get('normalized_phase',a['phase']);after=a.get('predicted_phase',before)
        band=a.get('calibration_ranges',{})
        local=band.get('local_limit_hz',300.);upper=band.get('global_high_hz',c['independent_end_hz'])
        low=a['min_usable_hz'];low_end=max(low*1.1,min(local*1.5,upper))
        intervals=[(max(low,local*.75),low_end,False),(max(low,local*.75),low_end,True),
            (max(low,upper*.6),min(a['f'][-1],upper*1.2),False),(max(low,upper*.6),min(a['f'][-1],upper*1.2),True)]
        for row,(lo,hi,derivative) in enumerate(intervals):
            if hi<=lo:lo,hi=low,min(a['f'][-1],max(low*1.2,upper))
            mask=(f>=lo)&(f<=hi);ax=axes[row,j]
            for curve,label,color in [(before,'校准前','#777'),(after,'实际系数预测','#0072B2')]:
                value=-np.gradient(curve,f)*1000/(2*np.pi) if derivative else np.rad2deg(curve)
                ax.plot(f[mask],value[mask],color=color,alpha=.25,lw=.8)
                ax.plot(f[mask],np.where(a['valid'][mask],value[mask],np.nan),color=color,label=label)
            ax.axhline(0,color='#555',lw=.6);ax.grid(alpha=.2);ax.legend()
            ax.set(xlim=(lo,hi),xlabel='Hz',ylabel='群延时 / ms' if derivative else '剩余相位 / °',title=a['name']+'：频段细节')
            ax.axvline(local if row<2 else upper,color='#888',ls='--',lw=.8,label='校准范围边界')
    fig.suptitle('频段细节 · 按实际范围显示，淡线为低可信区',fontsize=15)
    fig.savefig(out/'band_detail.png',dpi=150);plt.close(fig)
