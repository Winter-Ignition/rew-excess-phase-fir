"""放大衔接区，同时保留完整峰值，不让图示范围掩盖边缘问题。"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def build(analyses,c,out):
    fig,axes=plt.subplots(4,len(analyses),figsize=(7*len(analyses),12),squeeze=False,layout='constrained')
    for j,a in enumerate(analyses):
        f=a['f'];before=a.get('normalized_phase',a['phase']);after=a.get('predicted_phase',before)
        for row,(lo,hi,derivative) in enumerate([(250,450,False),(250,450,True),(6000,12000,False),(6000,12000,True)]):
            mask=(f>=lo)&(f<=hi);ax=axes[row,j]
            for curve,label,color in [(before,'校准前','#777'),(after,'实际系数预测','#0072B2')]:
                value=-np.gradient(curve,f)*1000/(2*np.pi) if derivative else np.rad2deg(curve)
                ax.plot(f[mask],value[mask],color=color,alpha=.25,lw=.8)
                ax.plot(f[mask],np.where(a['valid'][mask],value[mask],np.nan),color=color,label=label)
            ax.axhline(0,color='#555',lw=.6);ax.grid(alpha=.2);ax.legend()
            ax.set(xlim=(lo,hi),xlabel='Hz',ylabel='群延时 / ms' if derivative else '剩余相位 / °',title=a['name']+'：衔接检查')
            ax.axvspan(300 if hi<1000 else 8000,400 if hi<1000 else 10000,color='#aaa',alpha=.12)
    fig.suptitle('衔接放大 · 淡线为低可信区，仅用于诊断',fontsize=15)
    fig.savefig(out/'transitions.png',dpi=150);plt.close(fig)
