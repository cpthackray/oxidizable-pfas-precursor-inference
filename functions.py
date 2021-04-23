import numpy as np
import pandas as pd

BIGNEG = -1e32
MINVAL, MAXVAL = -6, 4

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
    0, 0.869, 0.0085, 0, 0, 0,
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
chains = [f'C{x}' for x in range(4, 11)]
fmeans = np.array([np.mean(ECFcomp[c]) for c in chains])
fstds = np.array([np.std(ECFcomp[c]) for c in chains])

def likelihood(x, meas, config):
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
    b = meas.b
    bpre = meas.bpre
    bpost = meas.bpost

    # print(bpre, bpost)
    mdls = meas.mdls
    berr = meas.berr
    C8 = meas.isC8

    x_p = 10**x[:-1]  # transform from log space
    # (last value is an error term)
    logprob = 0

    A, U = config.model, config.uncertainty

    mm = np.dot(A, x_p)
    uu = np.dot(U, x_p)
    mod = np.log10(mm)

    # errors
    e_p = x[-1] # error parameter
    moderr = (uu / mm)  # fractional error
    obserr = berr
    toterr = (moderr**2 + obserr**2 + e_p**2)**0.5

    # concentrations by difference
    if ((bpre is not None) and (bpost is not None)):
        for i in range(len(bpre)):
            delta = bpost[i] - bpre[i]
            rsd = bpre[i]*berr[i]
            if delta <= rsd:
                # change in PFCA concentration is indistinguishable from experimental error
                obsmin = MINVAL  # don't want -inf
                obsmax = np.log10(rsd)
                if obsmin <= mod[i] < obsmax:
                    logprob += 0
                elif mod[i] > obsmax:
                    logprob += -((mod[i] - obsmax) / toterr[i])**2
                else:
                    logprob += BIGNEG
            else:
                obs = np.log10(delta)
                logprob += -((mod[i] - obs) / toterr[i])**2

    else:
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


def prior_AFFF(x, meas, config):
    """Prior log-probability of proposal x.

    Prior used for AFFF samples.

    Arguments:
    x (array) proposal precursors

    Keyword arguments:
    PFOS (tuple(float,float)) observed PFOS and PFOS MDL

    Returns:
    (float) log-prior
    """
    PFOS2 = meas.pfos
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
        logprob += 0
    else:
        logprob += BIGNEG

    if emin < e_p < emax:
        logprob += 0
    else:
        logprob += BIGNEG


    # Evaluate the composition of ECF precursor proposal against their
    # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
    # the oxidation yields of ECF precursors do not depend on their
    # perfluorinated chain length (n)
    ecf_comp = x_p[3:] / ecf
    logprob += -(np.sum(np.abs((ecf_comp - config.compmeans) / (2 * config.compstds))**2))

    for i, xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            logprob += BIGNEG
        if xi > MAXVAL:  # or high
            logprob += BIGNEG
    if e_p < MINVAL:
        logprob += BIGNEG

    return logprob

def prior_AFFF_impacted(x, meas, config):
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
    PFOS2 = meas.pfos
    logprob = 0
    PFOS, MDL = PFOS2

    emin, emax = 0, 2

    x_p = 10**x[:-1]
    e_p = x[-1]
    ecf = np.sum(x_p[config.ecf_indices])  # sum only the ECF precursors

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
        logprob += 0
    else:
        logprob += BIGNEG

    if emin < e_p < emax:
        logprob += 0
    else:
        logprob += BIGNEG


    # Evaluate the composition of ECF precursor proposal against their
    # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
    # the oxidation yields of ECF precursors do not depend on their
    # perfluorinated chain length (n)
    ecf_comp = x_p[config.ecf_indices] / ecf
    logprob += -(np.sum(np.abs((ecf_comp - config.compmeans) / (config.compstds))**2))

    for i, xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            logprob += BIGNEG
        if xi > MAXVAL:  # or high
            logprob += BIGNEG
    if e_p < MINVAL:
        logprob += BIGNEG

    return logprob

def prior_unknown(x, meas, config):
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

    b = meas.b
    bpre = meas.bpre
    bpost = meas.bpost

    if (bpre is not None) and (bpost is not None):
        meassum = (bpost-bpre).sum()
    else:
        meassum = b.sum()

    x_p = 10**x[:-1]
    totp = np.sum(x_p)

    # Prevent inference from infering solutions with more than 10x the
    # measured mass
    if 1 < totp/meassum<10:
        cost += 0
    else:
        cost += BIGNEG

    for i,xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            cost += -BIGNEG
        if xi > MAXVAL: # or high
            cost += -BIGNEG
    return cost


# Collect priors by name for easy lookup
priors = {'AFFF': prior_AFFF,'AFFF_impacted':prior_AFFF_impacted,'unknown':prior_unknown}
