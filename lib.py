import numpy as np
import pandas as pd

BIGNEG = -1e32
MINVAL, MAXVAL = -2, 7

# TOP assay yields of representative n:2 FT and ECF precursors reported in the literature:
# n:2 FT precursors: 4:2 FTS, 5:3 FTCA, 6:2 FTA, 6:2 FTAB, 6:2 diPAP, 8:2 diPAP, 7:3 FTCA, 8:2 FTS, 10:2 FTS
# ECF precursors: FBSA, FHxSA, PFHxSAm, PFHxSAmS, FOSA, MeFOSA, EtFOSA, FOSAA, N-MeFOSAA, N-EtFOSAA,
# PFOAB, PFOSB, PFOANA, PFOSNO, PFOSAmS, PFOSAm, FDSA

# Average of n:2 FTs reported in Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T;
# Simonnet-Laprade et al., 2019, ESPI; Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T;
# and internal Heidi TOP fish analysis (20210222, 20210303)
# x_ft = [n, n-1, n-2, n-3, n-4, n-5, n-6, n-7]
x_ft = [0.0568, 0.2290, 0.3469, 0.2094, 0.1336, 0.0811, 0.0200, 0.0100]

# Standard deviation of average n:2 FTs reported in Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T;
# Simonnet-Laprade et al., 2019, ESPI; Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T;
# and internal Heidi TOP fish analysis (20210222, 20210303)
err_ft = [0.0586, 0.1100, 0.0986, 0.0488, 0.0330, 0.0468, 0.0212, 0.0141]

# Average of ECF precursors reported in Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T;
# Simonnet-Laprade et al., 2019, ESPI; Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T;
# Janda et al., 2019, ESPI; and internal Bridger (PFHxSAm/S) and Heidi TOP fish analyses (20210222, 20210303, 20210323)
# x_ecf = [n, n-1, n-2, n-3, n-4, n-5, n-6, n-7]
x_ecf = [0.0043, 0.8620, 0.0262, 0.0260, 0.0061, 0.0011, 0.0900, 0]

# Standard deviation of average ECF precursors reported in Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T;
# Simonnet-Laprade et al., 2019, ESPI; Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T;
# Janda et al., 2019, ESPI; and internal Bridger (PFHxSAm/S) and Heidi TOP fish analyses (20210222, 20210303, 20210323)
err_ecf = [0.0126, 0.1600, 0.0338, 0.0645, 0.0226, 0.0058, 0, 0]

# Prior information: the ratio of PFOS to ECF precursors in ECF AFFF
# From Houtz et al. 2013 Table S5 and S6
ECFcomp = pd.read_csv('data/3M_AFFF_Compositions.csv')
chains = [f'C{x}' for x in range(4, 11)]
fmeans = np.array([np.mean(ECFcomp[c]) for c in chains])
fstds = np.array([np.std(ECFcomp[c]) for c in chains])

precursor_order = ['4:2 FT',  '5:3 FT',  '6:2 FT',  '7:3 FT',  '8:2 FT',  '9:3 FT',  '10:2 FT',
                   'C4 ECF',  'C5 ECF',  'C6 ECF',  'C7 ECF',  'C8 ECF',  'C9 ECF',  'C10 ECF']
terminal_order = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'] 
def makeA():
    """ TOP assay PFCA yield matrix.

    Takes yields reported Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T; Simonnet-Laprade et al., 2019, ESPI;
    Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T; Janda et al., 2019, ESPI; and internal Bridger (PFHxSAm/S)
    and Heidi TOP fish analyses (20210222, 20210303, 20210323)
    and returns 8x14 matrices representing A and U from matrix eq (A±U)x=b """
    #       0       1       2       3       4       5       6       7       8       9       10      11      12      13
    #    4:2 FT  5:3 FT  6:2 FT  7:3 FT  8:2 FT  9:3 FT  10:2 FT  C4 ECF  C5 ECF  C6 ECF  C7 ECF  C8 ECF  C9 ECF  C10 ECF
   # A = [x1_FT, x2_FT,  x3_FT,  x4_FT,  x5_FT,  x6_FT,  x7_FT,   x1_ECF, x2_ECF, x3_ECF, x4_ECF, x5_ECF, x6_ECF, x7_ECF] #to C3  (PFBA)
   #     [x0_FT, x1_FT,  x2_FT,  x3_FT,  x4_FT,  x5_FT,  x6_FT,   x0_ECF, x1_ECF, x2_ECF, x3_ECF, x4_ECF, x5_ECF, x6_ECF] #to C4  (PFPeA)
   #     [0    , x0_FT,  x1_FT,  x2_FT,  x3_FT,  x4_FT,  x5_FT,   0     , x0_ECF, x1_ECF, x2_ECF, x3_ECF, x4_ECF, x5_ECF] #to C5  (PFHxA)
   #     [0    , 0    ,  x0_FT,  x1_FT,  x2_FT,  x3_FT,  x4_FT,   0     , 0     , x0_ECF, x1_ECF, x2_ECF, x3_ECF, x4_ECF] #to C6  (PFHpA)
   #     [0    , 0    ,  0    ,  x0_FT,  x1_FT,  x2_FT,  x3_FT,   0     , 0     , 0     , x0_ECF, x1_ECF, x2_ECF, x3_ECF] #to C7  (PFOA)
   #     [0    , 0    ,  0    ,  0    ,  x0_FT,  x1_FT,  x2_FT,   0     , 0     , 0     , 0     , x0_ECF, x1_ECF, x2_ECF] #to C8  (PFNA)
   #     [0    , 0    ,  0    ,  0    ,  0    ,  x0_FT,  x1_FT,   0     , 0     , 0     , 0     , 0     , x0_ECF, x1_ECF] #to C9  (PFDA)
   #     [0    , 0    ,  0    ,  0    ,  0    ,  0    ,  x0_FT,   0     , 0     , 0     , 0     , 0     , 0     , x0_ECF] #to C10 (PFUnDA)

    # U has the same structure as A, but is populated by
    # standard deviation (err) instead of mean

    # Construct A and U separately
    A = np.zeros((8, 14))
    U = np.zeros_like(A)

    # FT
    A[1, 0] = A[2, 1] = A[3, 2] = A[4, 3] = A[5, 4] = A[6, 5] = A[7, 6] = x_ft[0]
    U[1, 0] = U[2, 1] = U[3, 2] = U[4, 3] = U[5, 4] = U[6, 5] = U[7, 6] = err_ft[0]
    A[0, 0] = A[1, 1] = A[2, 2] = A[3, 3] = A[4, 4] = A[5, 5] = A[6, 6] = x_ft[1]
    U[0, 0] = U[1, 1] = U[2, 2] = U[3, 3] = U[4, 4] = U[5, 5] = U[6, 6] = err_ft[1]
    A[0, 1] = A[1, 2] = A[2, 3] = A[3, 4] = A[4, 5] = A[5, 6] = x_ft[2]
    U[0, 1] = U[1, 2] = U[2, 3] = U[3, 4] = U[4, 5] = U[5, 6] = err_ft[2]
    A[0, 2] = A[1, 3] = A[2, 4] = A[3, 5] = A[4, 6] = x_ft[3]
    U[0, 2] = U[1, 3] = U[2, 4] = U[3, 5] = U[4, 6] = err_ft[3]
    A[0, 3] = A[1, 4] = A[2, 5] = A[3, 6] = x_ft[4]
    U[0, 3] = U[1, 4] = U[2, 5] = U[3, 6] = err_ft[4]
    A[0, 4] = A[1, 5] = A[2, 6] = x_ft[5]
    U[0, 4] = U[1, 5] = U[2, 6] = err_ft[5]
    A[0, 5] = A[1, 6] = x_ft[6]
    U[0, 5] = U[1, 6] = err_ft[6]
    A[0, 6] = x_ft[7]
    U[0, 6] = err_ft[7]

    # ECF
    A[1, 7] = A[2, 8] = A[3, 9] = A[4, 10] = A[5, 11] = A[6, 12] = A[7, 13] = x_ecf[0]
    U[1, 7] = U[2, 8] = U[3, 9] = U[4, 10] = U[5, 11] = U[6, 12] = U[7, 13] = err_ecf[0]
    A[0, 7] = A[1, 8] = A[2, 9] = A[3, 10] = A[4, 11] = A[5, 12] = A[6, 13] = x_ecf[1]
    U[0, 7] = U[1, 8] = U[2, 9] = U[3, 10] = U[4, 11] = U[5, 12] = U[6, 13] = err_ecf[1]
    A[0, 8] = A[1, 9] = A[2, 10] = A[3, 11] = A[4, 12] = A[5, 13] = x_ecf[2]
    U[0, 8] = U[1, 9] = U[2, 10] = U[3, 11] = U[4, 12] = U[5, 13] = err_ecf[2]
    A[0, 9] = A[1, 10] = A[2, 11] = A[3, 12] = A[4, 13] = x_ecf[3]
    U[0, 9] = U[1, 10] = U[2, 11] = U[3, 12] = U[4, 13] = err_ecf[3]
    A[0, 10] = A[1, 11] = A[2, 12] = A[3, 13] = x_ecf[4]
    U[0, 10] = U[1, 11] = U[2, 12] = U[3, 13] = err_ecf[4]
    A[0, 11] = A[1, 12] = A[2, 13] = x_ecf[5]
    U[0, 11] = U[1, 12] = U[2, 13] = err_ecf[5]
    A[0, 12] = A[1, 13] = x_ecf[6]
    U[0, 12] = U[1, 13] = err_ecf[6]
    A[0, 13] = x_ecf[7]
    U[0, 13] = err_ecf[7]

    return (A, U)

def makeb(meas, C8=True):
    """Make the measurement array b.

    b is a 6x1 array of measurements C3-C8 if C8=True
    else, b is a 5x1 array of measurements C3-C7

    Arguments:
    meas (array) measurement values

    Keyword arguments:
    C8 (boolean) whether C8 was measured.

    Returns:
    (array) measurement array
    """
    if C8:
        return meas
    else:
        return meas[:5]

class Config(object):
    def __init__(self, filename):
        self.possible_precursors = precursor_order
        A,U = makeA()
        self.full_model = A
        self.full_uncertainty = U
        self.measured = None
        self.model = None
        self.uncertainty = None
        self.compmeans = None
        self.compstds = None
        
    def _set_measured(self, measured_list):
        self.measured = measured_list

    def _set_precursors(self, precursor_list):
        self.possible_precursors = precursor_list
        
    def _subset_model(self):
        choosep = []
        for p in self.possible_precursors:
            choosep.append(precursor_order.index(p))
        cp = np.array(sorted(choosep))
        choosec = []
        for c in self.measured:
            choosec.append(terminal_order.index(c))
        cc = np.array(sorted(choosec))
        self.model = self.full_model[:,cp][cc,:]
        self.uncertainty = self.full_uncertainty[:,cp][cc,:]
        
    def _set_ecf_ft_indices(self):
        ecf_indices = []
        ft_indices = []
        compmeans = []
        compstds = []
        for i,p in enumerate(self.possible_precursors):
            if 'ECF' in p:
                ecf_indices.append(i)
                chain = p[:-4]
                compmeans.append(fmeans[chains.index(chain)])
                std = fstds[chains.index(chain)]
                if std < 1e-5:
                    std = 1
                compstds.append(std)
            if 'FT' in p:
                ft_indices.append(i)
        self.ecf_indices = np.array(ecf_indices)
        self.ft_indices = np.array(ft_indices)
        self.compmeans = np.array(compmeans)
        self.compstds = np.array(compstds)
        
    def setup_model(self, measured_list, precursor_list=None):
        if precursor_list is None:
            precursor_list = precursor_order
        self._set_precursors(precursor_list)
        self._set_measured(measured_list)
        self._subset_model()
        self._set_ecf_ft_indices()
        
class Measurements(object):
    def __init__(self, b=None, bpre=None, bpost=None,
                 pfos=None, C8=False, mdls=None, errs=None):
        self.b = b
        self.bpre = bpre
        self.bpost = bpost
        self.pfos = pfos
        self.isC8 = C8
        self.mdls = mdls
        self.errs = errs
        self.whats_measured = ['C3', 'C4', 'C5', 'C6', 'C7']
        self.configfile = 'config.yaml'
        
    def from_row(self, dfrow):

        measurements = dfrow[['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'PFOS']].values
        try:
            premeas = dfrow[['C3pre', 'C4pre', 'C5pre', 'C6pre',
                      'C7pre', 'C8pre', 'PFOSpre']].values
        except KeyError:
            print("key error")
            premeas = None
        try:
            postmeas = dfrow[['C3post', 'C4post', 'C5post', 'C6post',
                              'C7post', 'C8post', 'PFOSpost']].values
        except KeyError:
            postmeas = None
        inmdlss = dfrow[['C3MDL', 'C4MDL', 'C5MDL', 'C6MDL', 'C7MDL', 'C8MDL',
                      'PFOSMDL']].values
        inobserrs = dfrow[['C3err', 'C4err', 'C5err', 'C6err', 'C7err', 'C8err',
                        'PFOSerr']].values
        nmeas = measurements.shape[0]
        measures = measurements
        mdlss = inmdlss
        errs = inobserrs

        try:
            cfg = dfrow['config']
        except:
            cfg = 'config.yaml'
        

        PFOS = (measures[6], mdlss[6])
        C8 = dfrow['C8incl']
        mdls = makeb(mdlss[:6], C8=C8)
        b = makeb(measures[:6], C8=C8)
        berr = makeb(errs[:6], C8=C8)
                
        self.b = b
        self.bpre = premeas
        self.bpost = postmeas
        self.mdls = mdls
        self.berr = berr
        self.isC8 = C8
        self.pfos = PFOS
        self.configfile = cfg
    
        
