'''
python -m cProfile -o <...> <script.py>
import pstats; p = pstats.Stats(...); p.sort_stats('time|cumulative|name').print_stats(10)
'''
import aipy
import aipy_cuda._aipy as a
import numpy as n
import optparse, sys
import time

o = optparse.OptionParser()
o.add_option('--nsim', dest='nsim', type='int', default=100,
    help='Number of simulations to run for each nsrc/nfreq.')
o.add_option('--nsrc_max', dest='nsrc_max', type='int', default=4096,
    help='Number of sources to simulate up to (A power of 2).')
o.add_option('--nfreq_max', dest='nfreq_max', type='int', default=32768,
    help='Number of frequencies to simulate up to (A power of 2).')
o.add_option('--nsrc_min', dest='nsrc_min', type='int', default=16,
    help='Starting number of sources to simulate (A power of 2).')
o.add_option('--nfreq_min', dest='nfreq_min', type='int', default=64,
    help='Starting number of frequencies to simulate (A power of 2).')
o.add_option('--no_gpu', dest='no_gpu', action='store_true',
    help='Use a software emulator of the GPU simulation code instead of the GPU.')
o.add_option('--o', dest='output', type='str', default='output',
    help='The output will be written to this file.')

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

i = opts.nsrc_min
nsrcs = [] #The list of nsrc values to iterate over
while i <= opts.nsrc_max:
    nsrcs.append(i)
    i = i*2

i = opts.nfreq_min
nfreqs = [] #The list of nfreq values to iterate over
while i <= opts.nfreq_max:
    nfreqs.append(i)
    i = i*2

for nsrc in nsrcs:
    for nfreq in nfreqs:
        baseline = n.array([100, 100, 0], dtype=n.float32)
        src_dir = n.array([[1.,0,0]] * nsrc, dtype=n.float32)
        src_int = n.array([10.] * nsrc, dtype=n.float32)
        src_index = n.array([0.] * nsrc, dtype=n.float32)
        freqs = n.linspace(.1,.2, nfreq).astype(n.float32)
        mfreqs = n.array([.150] * nsrc, dtype=n.float32)

        if not opts.no_gpu: vis_sim = a.vis_sim
        start = time.clock()
        for i in xrange(opts.nsim):
            print i,'/',opts.nsim
            vis = vis_sim(baseline, src_dir, src_int, src_index, freqs, mfreqs)
        elapsed = time.clock() - start
        with open(opts.output,'a') as f:
            f.write('Nfreqs = {0}\n Nsrc = {1}\n Time = {2}\n \n'.format(nfreq, nsrc, elapsed))

