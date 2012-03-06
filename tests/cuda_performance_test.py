'''
python -m cProfile -o <...> <script.py>
import pstats; p = pstats.Stats(...); p.sort_stats('time|cumulative|name').print_stats(10)
'''
import aipy
import aipy_cuda._aipy as a
import numpy as n
import optparse, sys

o = optparse.OptionParser()
o.add_option('--nsim', dest='nsim', type='int', default=100,
    help='Number of simulations to run.')
o.add_option('--nsrc', dest='nsrc', type='int', default=100,
    help='Number of sources to simulate.')
o.add_option('--nfreq', dest='nfreq', type='int', default=1024,
    help='Number of frequencies to simulate.')
o.add_option('--no_gpu', dest='no_gpu', action='store_true',
    help='Use a software emulator of the GPU simulation code instead of the GPU.')
opts,args = o.parse_args(sys.argv[1:])

def vis_sim(baseline, src_dir, src_int, src_index, freqs, mfreq):
    bl = baseline.copy(); bl.shape = (bl.size,1)
    bl = n.dot(src_dir, bl)
    freqs = freqs.copy(); freqs.shape = (1,freqs.size)
    src_int = src_int.copy(); src_int.shape = (src_int.size,1)
    src_index = src_index.copy(); src_index.shape = (src_index.size,1)
    mfreq = mfreq.copy(); mfreq.shape = (mfreq.size,1)
    amp = src_int * (freqs/mfreq)**src_index
    phs = n.exp(-2j*n.pi * freqs * bl)
    return n.sum(amp * phs, axis=0)

opts.nfreq = 1024
opts.nsrc = 100
opts.nsim = 100

baseline = n.array([100, 100, 0], dtype=n.float32)
src_dir = n.array([[1.,0,0]] * opts.nsrc, dtype=n.float32)
src_int = n.array([10.] * opts.nsrc, dtype=n.float32)
src_index = n.array([0.] * opts.nsrc, dtype=n.float32)
freqs = n.linspace(.1,.2, opts.nfreq).astype(n.float32)
mfreqs = n.array([.150] * opts.nsrc, dtype=n.float32)

if not opts.no_gpu: vis_sim = a.vis_sim

for i in xrange(opts.nsim):
    print i,'/',opts.nsim
    vis = vis_sim(baseline, src_dir, src_int, src_index, freqs, mfreqs)

