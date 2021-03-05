import numpy as np
import pandas as pd

BIGNEG = -1e32
MINVAL, MAXVAL = -2, 7

# TOP assay yields of representative n:2 FT and ECF precursors
# reported in the literature
# Average of n:2 FTs reported in Houtz and Sedlak Table 1, 2012
# and Martin et al. 2019 Table 1
x_ft = [
    0.0245, 0.1850, 0.2874, 0.1943, 0.1425, 0.0867
]
# Standard deviation of average n:2 FTs reported in Houtz
# and Sedlak Table 1, 2012 and Martin et al. 2019 Table 1
err_ft = [
    0.0065, 0.0721, 0.0435, 0.0207, 0.0171, 0.0252
]
# Average of ECF precursors reported in Houtz and Sedlak 2012 Table 1,
# Martin et al. 2019 Table 1, Janda et al. 2019 Table 1,
# and internal data of PFHxSAm (88% yield to Cn-1)
# and PFHxSAmS (87% yield to Cn-1)
x_ecf = [
    0, 0.869, 0.0085, 0, 0, 0
]
# Standard deviation of average ECF precursors reported in Houtz
# and Sedlak 2012 Table 1, Martin et al. 2019 Table 1, Janda et al. 2019
# Table 1, and internal data of PFHxSAm (88% yield to Cn-1)
# and PFHxSAmS (87% yield to Cn-1)
err_ecf = [
    0, 0.1144, 0.0089, 0, 0, 0
]

# Prior information: the ratio of PFOS to ECF precursors in ECF AFFF
# From Houtz et al. 2013 Table S5 and S6
ECFcomp = pd.read_csv('data/3M_AFFF_Compositions.csv')
chains = [f'C{x}' for x in range(4, 9)]
fmeans = np.array([np.mean(ECFcomp[c]) for c in chains])
fstds = np.array([np.std(ECFcomp[c]) for c in chains])

def makeA():
    """ TOP assay PFCA yield matrix.

    Takes yields reported in Houtz and Sedlack 2012,
    Martin et al. 2019, Janda et al. 2019, and internal and returns
    6x8 matrices representing A and U from matrix eq (A±U)x=b
    """

    #      4:2 ft  6:2 ft  8:2 ft  C4 ECF  C5 ECF  C6 ECF  C7 ECF  C8 ECF
    # A = [x1_FT,  x3_FT,  x5_FT,  x1_ECF, x2_ECF, x3_ECF, x4_ECF, x5_ECF] #to C3 (PFBA)
    #     [x0_FT,  x2_FT,  x4_FT,  x0_ECF, x1_ECF, x2_ECF, x3_ECF, x4_ECF] #to C4 (PFPeA)
    #     [0    ,  x1_FT,  x3_FT,  0     , x0_ECF, x1_ECF, x2_ECF, x3_ECF] #to C5 (PFHxA)
    #     [0    ,  x0_FT,  x2_FT,  0     , 0     , x0_ECF, x1_ECF, x2_ECF] #to C6 (PFHpA)
    #     [0    , 0     ,  x1_FT,  0     , 0      , 0    , x0_ECF, x1_ECF] #to C7 (PFOA)
    #     [0    , 0     ,  x0_FT,  0     , 0      , 0    , 0     , x0_ECF] #to C8 (PFNA)

    # U has the same structure as A, but is populated by
    # standard deviation (err) instead of mean

    # Construct A and U separately
    A = np.zeros((6, 8))
    U = np.zeros_like(A)

    # FT
    A[1, 0] = A[3, 1] = A[5, 2] = x_ft[0]
    U[1, 0] = U[3, 1] = U[5, 2] = err_ft[0]
    A[0, 0] = A[2, 1] = A[4, 2] = x_ft[1]
    U[0, 0] = U[2, 1] = U[4, 2] = err_ft[1]
    A[1, 1] = A[3, 2] = x_ft[2]
    U[1, 1] = U[3, 2] = err_ft[2]
    A[0, 1] = A[2, 2] = x_ft[3]
    U[0, 1] = U[2, 2] = err_ft[3]
    A[1, 2] = x_ft[4]
    U[1, 2] = err_ft[4]
    A[0, 2] = x_ft[5]
    U[0, 2] = err_ft[5]

    # ECF
    A[1, 3] = A[2, 4] = A[3, 5] = A[4, 6] = A[5, 7] = x_ecf[0]
    U[1, 3] = U[2, 4] = U[3, 5] = U[4, 6] = U[5, 7] = err_ecf[0]
    A[0, 3] = A[1, 4] = A[2, 5] = A[3, 6] = A[4, 7] = x_ecf[1]
    U[0, 3] = U[1, 4] = U[2, 5] = U[3, 6] = U[4, 7] = err_ecf[1]
    A[0, 4] = A[1, 5] = A[2, 6] = A[3, 7] = x_ecf[2]
    U[0, 4] = U[1, 5] = U[2, 6] = U[3, 7] = err_ecf[2]
    A[0, 5] = A[1, 6] = A[2, 7] = x_ecf[3]
    U[0, 5] = U[1, 6] = U[2, 7] = err_ecf[3]
    A[0, 6] = A[1, 7] = x_ecf[4]
    U[0, 6] = U[1, 7] = err_ecf[4]
    A[0, 7] = x_ecf[5]
    U[0, 7] = err_ecf[5]

    return (A, U)


def likelihood(x, b, mdls, berr, C8):
    """log-likelihood of x given b.

    Calculates the log-probability of x given observations b,
    with additional considerations based on the MDLs and
    whether C8 (PFNA) was measured.

    Arguments:
    x (array) simulated proposal of precursors
    b (array) observations of products (C3-C7 PFCA)
    mdls (array) MDLs of product observations
    berr (array) relative errors of observations
    C8 (boolean) whether C8 (PFNA) was measured

    Returns:
    (float) log-likelihood
    """
    x_p = 10**x[:-1]  # transform from log space

    logprob = 0
    A, U = makeA()  # take a random sample of the matrix
    if not C8:
        A = A[:-1, :]
        U = U[:-1, :]
    mm = np.dot(A, x_p)
    uu = np.dot(U, x_p)
    mod = np.log10(mm)

    # errors
    e_p = x[-1]
    moderr = (uu / mm)  # fractional error
    obserr = berr
    toterr = (moderr**2 + obserr**2 + e_p**2)**0.5

    for i in range(len(b)):
        if b[i] <= mdls[i]:
            # non-detect
            obsmin = MINVAL  # don't want -inf
            obsmax = np.log10(mdls[i] / np.sqrt(2))
            if obsmin <= mod[i] < obsmax:
                logprob += 0
            elif mod[i] > obsmax:
                logprob += -((mod[i] - obsmax) / toterr[i])**2
            else:
                logprob += BIGNEG
        else:
            obs = np.log10(b[i])
            logprob += -((mod[i] - obs) / toterr[i])**2

    if np.isnan(logprob):
        print(x_p, A)
    return logprob


def prior_AFFF(x, **kwargs):
    """Prior log-probability of proposal x.

    Prior used for AFFF samples.

    Arguments:
    x (array) proposal precursors

    Keyword arguments:
    PFOS (tuple(float,float)) observed PFOS and PFOS MDL

    Returns:
    (float) log-prior
    """
    PFOS2 = kwargs.get('PFOS')
    logprob = 0
    PFOS, MDL = PFOS2

    emin, emax = 0, 2

    x_p = 10**x[:-1]
    e_p = x[-1]
    ecf = np.sum(x_p[3:])  # sum only the ECF precursors

    # the sum of ECF precursors should fall inside of lower and upper bounds
    # of the ratio of TOP assay precursors (corrected for their PFCA yield
    # [88±12%]) to PFOS reported in 3M AFFF in Houtz et al. Tables S5 and S6
    lowratio = 0.84
    highratio = 2.73
    if PFOS < MDL:
        PFOS = MDL / np.sqrt(2)
        lowratio = 0
    pmin, pmax = PFOS * lowratio, PFOS * highratio

    if pmin < ecf < pmax:
        logprob = 0
    else:
        logprob = BIGNEG

    if emin < e_p < emax:
        logprob += 0
    else:
        logprob += BIGNEG


    # Evaluate the composition of ECF precursor proposal against their
    # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
    # the oxidation yields of ECF precursors do not depend on their
    # perfluorinated chain length (n)
    ecf_comp = x_p[3:] / ecf
    logprob += -(np.sum(np.abs((ecf_comp - fmeans) / (2 * fstds))**2))

    for i, xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            logprob += BIGNEG
        if xi > MAXVAL:  # or high
            logprob += BIGNEG
    if e_p < MINVAL:
        logprob += BIGNEG

    return logprob

def prior_AFFF_impacted(x, **kwargs):
    """Prior log-probability of proposal x.

    Prior used for environmental samples where predominant PFAS source
    is thought to be AFFF.

    Arguments:
    x (array) proposal precursors

    Keyword arguments:
    PFOS (tuple(float,float)) observed PFOS and PFOS MDL

    Returns:
    (float) log-prior
    """
    PFOS2 = kwargs.get('PFOS')
    logprob = 0
    PFOS, MDL = PFOS2

    emin, emax = 0, 2

    x_p = 10**x[:-1]
    e_p = x[-1]
    ecf = np.sum(x_p[3:])  # sum only the ECF precursors

    # the sum of ECF precursors should fall inside of lower and upper bounds
    # of the ratio of TOP assay precursors (corrected for their PFCA yield
    # [88±12%]) to PFOS reported in 3M AFFF in Houtz et al. Tables S5 and S6
    lowratio = 0.84
    highratio = 2.73
    if PFOS < MDL:
        PFOS = MDL / np.sqrt(2)
        lowratio = 0
    pmin, pmax = PFOS * lowratio, PFOS * highratio

    if pmin < ecf < pmax:
        logprob = 0
    else:
        logprob = BIGNEG

    if emin < e_p < emax:
        logprob += 0
    else:
        logprob += BIGNEG


    # Evaluate the composition of ECF precursor proposal against their
    # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
    # the oxidation yields of ECF precursors do not depend on their
    # perfluorinated chain length (n)
    ecf_comp = x_p[3:] / ecf
    logprob += -(np.sum(np.abs((ecf_comp - fmeans) / (fstds))**2))

    for i, xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            logprob += BIGNEG
        if xi > MAXVAL:  # or high
            logprob += BIGNEG
    if e_p < MINVAL:
        logprob += BIGNEG

    return logprob

def prior_unknown(x, **kwargs):
    """Prior log-probability of proposal x.

    Prior used for environmental samples where predominant PFAS source
    is unknown.

    Arguments:
    x (array) proposal precursors

    Keyword arguments:
    b (array of floats) measured molar increases in PFCA from TOP assay

    Returns:
    (float) log-prior
    """

    cost = 0

    b = kwargs.get('b')
    meassum = b.sum()

    x_p = 10**x
    totp = np.sum(x_p)

    # Prevent inference from infering solutions with more than 10x the
    # measured mass
    if 1 < totp/meassum<10:
        cost += 0
    else:
        cost += -1e32

    for i,xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            cost += -1e32
        if xi > MAXVAL: # or high
            cost += -1e32
    return cost

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

# Collect priors by name for easy lookup
priors = {'AFFF': prior_AFFF,'AFFF_impacted':prior_AFFF_impacted,'unknown':prior_unknown}
