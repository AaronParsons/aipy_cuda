import aipy
import aipy_cuda._aipy as a
import numpy as n, unittest

def interp3d(data, x,y,z, xlim, ylim, zlim):
    assert(data.ndim == 3)
    x0,x1 = map(float, xlim)
    y0,y1 = map(float, ylim)
    z0,z1 = map(float, zlim)
    dx = (x1-x0) / (data.shape[0]-1) # XXX or is it data.shape[0]
    dy = (y1-y0) / (data.shape[1]-1) # XXX or is it data.shape[0]
    dz = (z1-z0) / (data.shape[2]-1) # XXX or is it data.shape[0]
    x_px = (x-x0)/dx
    y_px = (y-y0)/dy
    z_px = (z-z0)/dz
    # XXX should probably do a bounds check here
    x_px_fl = n.floor(x_px).astype(n.int)
    y_px_fl = n.floor(y_px).astype(n.int)
    z_px_fl = n.floor(z_px).astype(n.int)
    d000 = data[x  ,y  ,z  ]
    d100 = data[x+1,y  ,z  ]
    d010 = data[x  ,y+1,z  ]
    d110 = data[x+1,y+1,z  ]
    d001 = data[x  ,y  ,z+1]
    d101 = data[x+1,y  ,z+1]
    d011 = data[x  ,y+1,z+1]
    d111 = data[x+1,y+1,z+1]
    x_fr = x_px - x_px_fl
    y_fr = y_px - y_px_fl
    z_fr = z_px - z_px_fl
    d = ((d000 * (1-x_fr) + d100 * x_fr) * (1-y_fr) + \
         (d010 * (1-x_fr) + d110 * x_fr) * y_fr   ) * (1-z_fr) + \
        ((d001 * (1-x_fr) + d101 * x_fr) * (1-y_fr) + \
         (d011 * (1-x_fr) + d111 * x_fr) * y_fr   ) * z_fr
    return d

class TestInterp3d(unittest.TestCase):
    def setUp(self):
        self.data_ones = n.ones((100,100,10), dtype=n.float32)
        self.xmin,self.xmax = (-1,1)
        self.ymin,self.ymax = (-1,1)
        self.zmin, self.zmax = (.100, .200) # GHz
    def test_ones(self):
        self.assertTrue(n.all(interp3d(self.data_ones, .1, .1, .1, (self.xmin,self.xmax), (self.ymin,self.ymax), (self.zmin,self.zmax))))
    

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

#def vis_sim(baseline, src_dir, src_int, src_index, freqs, mfreq,
#        beam_arr, lmin, lmax, mmin, max, beamfqmin, beamfqmaxq):
#    bl = baseline.copy(); bl.shape = (bl.size,1)
#    bl = n.dot(src_dir, bl)
#    freqs = freqs.copy(); freqs.shape = (1,freqs.size)
#    src_int = src_int.copy(); src_int.shape = (src_int.size,1)
#    src_index = src_index.copy(); src_index.shape = (src_index.size,1)
#    mfreq = mfreq.copy(); mfreq.shape = (mfreq.size,1)
#    amp = src_int * (freqs/mfreq)**src_index
#    phs = n.exp(-2j*n.pi * freqs * bl)
#    return n.sum(amp * phs, axis=0)

class Test_Aipy(unittest.TestCase):
    def setUp(self):
        self.NFREQ = 32
        self.baseline = n.array([100, 100, 0], dtype=n.float32)
        self.src_dir = n.array([[1.,0,0],[0, 1., 0]], dtype=n.float32)
        self.src_int = n.array([10., 100], dtype=n.float32)
        self.src_index = n.array([0., 0], dtype=n.float32)
        self.freqs = n.linspace(.1,.2, self.NFREQ).astype(n.float32)
        self.mfreqs = n.array([.150] * 2, dtype=n.float32)
        self.beam_arr = n.ones((100,100,10), dtype=n.float32)
        self.lmin, self.lmax = (-1,1)
        self.mmin, self.mmax = (-1,1)
        self.beamfqmin, self.beamfqmax = (.100, .200) # GHz
        self.vis = vis_sim(self.baseline, self.src_dir[:1], self.src_int[:1], self.src_index[:1], self.freqs, self.mfreqs[:1])
    def test_beam_arr_type(self):
        beam_arr = n.ones((100,100), dtype=n.float32)
        self.assertRaises(ValueError, a.vis_sim,
           self.baseline, self.src_dir, self.src_int, self.src_index, self.freqs, self.mfreqs,
                beam_arr, self.lmin, self.lmax, self.mmin, self.mmax, self.beamfqmin, self.beamfqmax)
        beam_arr = n.ones((100,100,10), dtype=n.int)
        self.assertRaises(ValueError, a.vis_sim,
           self.baseline, self.src_dir, self.src_int, self.src_index, self.freqs, self.mfreqs,
                beam_arr, self.lmin, self.lmax, self.mmin, self.mmax, self.beamfqmin, self.beamfqmax)
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
