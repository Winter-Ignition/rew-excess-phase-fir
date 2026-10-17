"""v2 回归测试，仅运行合成声学与系数，不包含个人测量。"""
from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from epfir.common import DEFAULTS,write_json,request,sha,minimum
from epfir.low_reference import reference,curve
from epfir.unified import continuation,loaded,half_difference
from epfir.analysis import analyze
from epfir.inputs import load
from phase_fir import run


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.c={**DEFAULTS,'delay_samples':8192,'head_half_ranges_cm':[5,5,5]}
    def tearDown(self):self.temp.cleanup()
    def test_noise_below_cutoff_cannot_move_model(self):
        f=np.fft.rfftfreq(131072,1/48000)
        mag=10**(curve(f,[0,np.log(32),0],4)/20);noisy=mag.copy();noisy[f<25]=.2
        a,am=reference(f,mag,25);b,bm=reference(f,noisy,25)
        np.testing.assert_allclose(a,b,atol=1e-12)
        self.assertEqual(am['models'],bm['models'])
        self.assertLess(am['models'][0]['rms_db'],.2)
    def test_frozen_model_changes_level_only(self):
        f=np.fft.rfftfreq(131072,1/48000);m=10**(curve(f,[0,np.log(35),0],4)/20)
        _,model=reference(f,m,25);_,other=reference(f,m*2,25,model)
        for a,b in zip(model['models'],other['models']):
            self.assertEqual(a['order'],b['order']);np.testing.assert_array_equal(a['parameters'][1:],b['parameters'][1:])
        self.assertAlmostEqual(other['models'][0]['parameters'][0]-model['models'][0]['parameters'][0],20*np.log10(2),delta=1e-4)
    def test_handoff_matches_value_slope_and_curvature(self):
        f=np.linspace(5000,23000,36001);y=.3+f*2e-5+(f-10000)**2*1e-9
        p=continuation(f,y,10000,21000);i=np.argmin(abs(f-10000));j=np.argmin(abs(f-21000))
        self.assertAlmostEqual(p[i],y[i],places=10)
        self.assertLess(abs((p[i+1]-p[i-1])/(f[i+1]-f[i-1])-2e-5),1e-8)
        self.assertLess(abs(p[j-1]),1e-8);self.assertEqual(p[j+1],0.)
        self.assertGreater(abs(p[np.argmin(abs(f-12000))]),.1)
    def test_cutoff_and_undistorted_limit_are_distinct(self):
        p=self.root/'x.wav';h=np.zeros(2048);h[200]=1;sf.write(p,h,48000,subtype='FLOAT')
        r=self.root/'request.json';write_json(r,dict(min_usable_hz=35,measurement_valid_low_hz=50,channels=[dict(name='X',path=str(p),kind='ir')]))
        c=request(r);self.assertEqual(c['channels'][0]['min_usable_hz'],50)
        write_json(r,dict(existing_phase_correction='this_filter',channels=[dict(name='X',path=str(p))]))
        with self.assertRaisesRegex(ValueError,'loaded_filters'):request(r)
    def test_dense_crop_allowed_only_with_noise_boundary(self):
        n=131072;f=np.fft.rfftfreq(n,1/48000)[1:];mask=f>=20
        p=self.root/'crop.txt';np.savetxt(p,np.c_[f[mask],np.zeros(mask.sum()),np.zeros(mask.sum())])
        ch=dict(path=str(p),kind='frd',measurement_sample_rate=48000,min_usable_hz=40)
        with self.assertRaisesRegex(ValueError,'边界'):load(ch,4194304)
        self.assertEqual(load({**ch,'measurement_valid_low_hz':25},4194304)['kind'],'frd')
    def test_replacement_contains_old_once_and_one_delay(self):
        n=131072;f=np.fft.rfftfreq(n,1/48000)
        # 复测为轻微剩余相位；已加载旧滤波自身具有不同、可分辨的相位。
        oldf=np.fft.rfftfreq(262144,1/192000)
        from epfir.design import render
        oldphase=.12*np.exp(-.5*(np.log(np.maximum(oldf,1)/1500)/.6)**2)
        h=render(oldf,oldphase,self.c);oldpath=self.root/'loaded.wav';sf.write(oldpath,h,192000,subtype='FLOAT')
        residual=.05*np.exp(-.5*(np.log(np.maximum(f,1)/700)/.8)**2)
        H=np.exp(1j*residual-2j*np.pi*f*.005);H[0]=1;H[-1]=H[-1].real
        measured=self.root/'measured.wav';sf.write(measured,np.fft.irfft(H,n),48000,subtype='FLOAT')
        r=self.root/'request.json';write_json(r,dict(channels=[dict(name='X',path=str(measured),kind='ir',crossover_hz=[1900])],
            existing_phase_correction='this_filter',head_half_ranges_cm=[5,5,5],
            loaded_filters=[dict(path=str(oldpath),sha256=sha(oldpath),channels=['X'],delay_samples=8192)]))
        out=self.root/'result';self.assertTrue(run('design',r,out)['passed'])
        m=json.loads((out/'manifest.json').read_text(encoding='utf-8'));self.assertEqual(m['role'],'complete_replacement')
        a=np.load(out/'analysis.npz');new,_=sf.read(out/m['files'][0]['file'])
        self.assertEqual(int(np.argmax(abs(new))),8192)
        ff=np.fft.rfftfreq(262144,1/192000);ratio=np.fft.rfft(new,262144)/np.fft.rfft(h,262144)
        target=np.interp(ff,a['X__f'],a['X__residual_target'],left=0,right=0)
        band=(ff>=100)&(ff<=10000)
        error=np.angle(ratio[band]*np.exp(-1j*target[band]));self.assertLess(np.rad2deg(abs(error)).max(),.2)
        wrong=(np.fft.rfft(new,262144)/np.exp(-2j*np.pi*ff*8192/192000))*np.exp(-1j*target)
        self.assertGreater(np.rad2deg(abs(np.angle(wrong[band]))).max(),4.)
        a.close()
    def test_loaded_filter_hash_and_delay_guard(self):
        p=self.root/'old.wav';h=np.zeros(16384);h[8192]=1;sf.write(p,h,192000,subtype='FLOAT')
        c={**self.c,'loaded_filters':[dict(path=str(p),sha256='wrong',channels=['X'])]}
        with self.assertRaisesRegex(ValueError,'变化'):loaded(dict(name='X'),c)
    def test_rounded_192k_export_does_not_accumulate_grid_error(self):
        n=524288;f=np.fft.rfftfreq(n,1/192000)[1:]
        p=self.root/'rounded.txt';np.savetxt(p,np.c_[f,np.zeros(len(f)),np.zeros(len(f))],fmt='%.6f',delimiter=',')
        source=load(dict(path=str(p),kind='frd',measurement_sample_rate=192000),4194304)
        self.assertTrue(source['meta']['grid_is_linear']);self.assertEqual(len(source['samples']),n)
    def test_low_frequency_turn_cannot_become_hf_half_turn(self):
        from epfir.model import ap
        f=np.geomspace(5,23000,3200);left=ap(f,70,.7)+2*np.pi
        delta=half_difference(f,left,np.zeros(len(f)))
        self.assertLess(abs(delta[f>=8000]).max(),.02)
    def test_noise_aware_extraction_on_known_minimum_phase_rolloff(self):
        n=131072;f=np.fft.rfftfreq(n,1/48000)
        mag=10**(curve(f,[0,np.log(32),0],4)/20)
        H=minimum(np.maximum(mag,1e-15),n)*np.exp(-2j*np.pi*f*233/48000)
        # 只污染明确无效的低端；有效带仍是已知最小相位真值。
        H[f<25]=.01*np.exp(1j*np.sin(f[f<25]*3));H[0]=.01
        p=self.root/'noisy.txt';np.savetxt(p,np.c_[f[1:],20*np.log10(np.maximum(abs(H[1:]),1e-15)),np.rad2deg(np.angle(H[1:]))])
        a=analyze(dict(name='X',path=str(p),kind='frd',measurement_sample_rate=48000,min_usable_hz=40,measurement_valid_low_hz=25),self.c)
        band=(a['f']>=40)&(a['f']<=300)&a['valid']
        self.assertLess(np.sqrt(np.mean(np.rad2deg(a['phase'][band])**2)),3.)
        self.assertFalse(a['smoothing_input_valid'][a['f']<25].any())


if __name__=='__main__':unittest.main(verbosity=2)
