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

    def from_row(self, dfrow):

        measurements = dfrow[['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'PFOS']].values
        try:
            premeas = dfrow[['C3pre', 'C4pre', 'C5pre', 'C6pre',
                      'C7pre', 'C8pre', 'PFOSpre']].values
        except KeyError:
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
    
        
