""" Generate samples of the posterior for PFAA precursors from TOP assay
measurements in aqueous matrices.

Authors:
Colin Thackray (thackray@seas.harvard.edu)
Bridger Ruyle (bruyle@g.harvard.edu)
"""
import argparse
import numpy as np
import pandas as pd
from sampling import sample_measurement
from functions import makeb

# Command line arguments
parser = argparse.ArgumentParser(
    description='Sample posterior for precursors.')
parser.add_argument('ISTART', metavar='istart', type=int,
                    help='first sample index (first index is 0)', default=0)
parser.add_argument('IEND', metavar='iend', type=int, nargs='?',
                    help='last sample index (first index is 0)', default=None)
parser.add_argument('-d', '--datafile', dest='FILENAME', action='store',
                    default='data/measurements.csv',
                    help='location of measurements file')
parser.add_argument('-o', '--outfile', dest='OUTFILE_STEM', action='store',
                    default='infer_out/mcmcout_',
                    help='Stem for output filename. Will gain suffix \
                    N (sample index)')
parser.add_argument('-t', '--target-steps', dest='TARGET',
                    action='store', default=2500, type=int,
                    help='Effective sample size to attain')
parser.add_argument('-m', '--max-steps', dest='MAX_STEPS',
                    action='store', default=50000, type=int,
                    help='Maximum number of steps before quiting.')
parser.add_argument('-D', '--max-depth', dest='MAX_DEPTH',
                    action='store', default=3, type=int,
                    help='Maximum depth of windowing in sampler tuning.')

args = parser.parse_args()
if args.IEND is None:
    args.IEND = args.ISTART

# Load input data from disk
df = pd.read_csv(args.FILENAME)
names = df['Sample'].values

measurements = df[['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'PFOS']].values
inmdlss = df[['C3MDL', 'C4MDL', 'C5MDL', 'C6MDL', 'C7MDL', 'C8MDL',
              'PFOSMDL']].values
inobserrs = df[['C3err', 'C4err', 'C5err', 'C6err', 'C7err', 'C8err',
                'PFOSerr']].values
nmeas = measurements.shape[0]
measures = measurements
mdlss = inmdlss
errs = inobserrs

# Do sampling for requested measurements
for bi in range(args.ISTART, args.IEND+1):
    print('Calculating for sample ' + df['Sample'][bi], end='')

    PFOS = (measures[bi, 6], mdlss[bi, 6])

    # Not everyone measures C8 (PFNA), but if measured should be used.
    C8 = df['C8incl'][bi]  # Boolean

    # Generate obs and MDL arrays
    mdls = makeb(mdlss[bi, :6], C8=C8)
    b = makeb(measures[bi, :6], C8=C8)
    berr = makeb(errs[bi, :6], C8=C8)

    prior_name = df['prior_name'][bi]
    print(f' with prior {prior_name}.')

    # Run MCMC ensemble to sample posterior
    sampler = sample_measurement(b,
                                 mdls,
                                 berr,
                                 PFOS,
                                 prior=prior_name,
                                 C8=C8,
                                 Nincrement=1000,
                                 TARGET_EFFECTIVE_STEPS=args.TARGET,
                                 MAX_STEPS=args.MAX_STEPS,
                                 MAX_DEPTH=args.MAX_DEPTH)

    # Save sampling output to disk
    trajectory = sampler.flatchain[:,:-1]
    outfile = f'{args.OUTFILE_STEM}{bi}'
    np.save(outfile, trajectory)
