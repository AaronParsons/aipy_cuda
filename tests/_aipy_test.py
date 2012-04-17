import aipy
import aipy_cuda._aipy as a
import numpy as n, unittest

def vis_sim(baseline, src_dir, src_int, src_index, freqs, mfreq): # XXX need to update to also use beam array
    bl = baseline.copy(); bl.shape = (bl.size,1)
    bl = n.dot(src_dir, bl)
    freqs = freqs.copy(); freqs.shape = (1,freqs.size)
    src_int = src_int.copy(); src_int.shape = (src_int.size,1)
    src_index = src_index.copy(); src_index.shape = (src_index.size,1)
    mfreq = mfreq.copy(); mfreq.shape = (mfreq.size,1)
    amp = src_int * (freqs/mfreq)**src_index
    phs = n.exp(-2j*n.pi * freqs * bl)
    return n.sum(amp * phs, axis=0)

class Test_Aipy(unittest.TestCase):
    def setUp(self):
        self.NFREQ = 32
        self.baseline = n.array([100, 100, 0], dtype=n.float32)
        self.src_dir = n.array([[1.,0,0],[0, 1., 0]], dtype=n.float32)
        self.src_int = n.array([10., 100], dtype=n.float32)
        self.src_index = n.array([0., 0], dtype=n.float32)
        self.freqs = n.linspace(.1,.2, self.NFREQ).astype(n.float32)
        self.mfreqs = n.array([.150] * 2, dtype=n.float32)
        self.beam_arr = n.ones((100,100,100), dtype=n.float32)
        self.lmin, self.lmax = (-1,1)
        self.mmin, self.mmax = (-1,1)
        self.beamfqmin, self.beamfqmax = (.100, .200) # GHz
        self.vis = vis_sim(self.baseline, self.src_dir[:1], self.src_int[:1], self.src_index[:1], self.freqs, self.mfreqs[:1])
    def test_vis_sim_return(self):
        vis = a.vis_sim(self.baseline, self.src_dir, self.src_int, self.src_index, self.freqs, self.mfreqs,
            self.beam_arr, self.lmin, self.lmax, self.mmin, self.mmax, self.beamfqmin, self.beamfqmax)
        self.assertEqual(vis.size, self.NFREQ)
        self.assertEqual(vis.dtype, n.complex64)
    def test_vis_1src_value(self):
        vis = a.vis_sim(self.baseline, self.src_dir[:1], self.src_int[:1], self.src_index[:1], self.freqs, self.mfreqs[:1],
            self.beam_arr, self.lmin, self.lmax, self.mmin, self.mmax, self.beamfqmin, self.beamfqmax)
        vis_ans = vis_sim(self.baseline, self.src_dir[:1], self.src_int[:1], self.src_index[:1], self.freqs, self.mfreqs[:1])
        #print vis
        #print vis_ans
        self.assertTrue(n.all(n.around(n.abs(vis - vis_ans), 3) == 0))
    def test_vis_2src_value(self):
        vis = a.vis_sim(self.baseline, self.src_dir, self.src_int, self.src_index, self.freqs, self.mfreqs,
            self.beam_arr, self.lmin, self.lmax, self.mmin, self.mmax, self.beamfqmin, self.beamfqmax)
        vis_ans = vis_sim(self.baseline, self.src_dir, self.src_int, self.src_index, self.freqs, self.mfreqs)
        self.assertTrue(n.all(n.around(n.abs(vis - vis_ans), 2) == 0))
        

if __name__ == '__main__':
    unittest.main()
