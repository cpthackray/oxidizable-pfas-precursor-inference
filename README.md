# Description
oxidizable-pfas-precursor-inference is a command line tool to infer concentrations of oxidizable
precursors aggregated by perfluorinated chain length (n) and manufacturing
origin (electrochemical fluorination: ECF vs fluorotelomerization: FT) based
on changes in perfluoroalkyl carboxylates (PFCA) in the total oxidizable
precursor (TOP) assay.

infer_precursors.py can be used to analyze TOP assay results for any aqueous
sample with the appropriate choice of a prior. This package
provides one built-in prior:
  * prior_AFFF – used for AFFF stocks ([Ruyle et al. 2020](http://dx.doi.org/10.1021/acs.estlett.0c00798))  

This prior can be used as a template and adapted for other specific purposes.

### Credit:
  * Co-authorship is appropriate if your paper benefits significantly from use
  of this model/code  
  * Citation is appropriate if use of this model/code has only a marginal impact
  on your work or if the work is a second generation application of the model/code.

This model was created by
[Colin P. Thackray](https://scholar.harvard.edu/thackray/about) and
[Bridger J. Ruyle](https://scholar.harvard.edu/ruyle) and originally
presented in [Ruyle et al. 2020](http://dx.doi.org/10.1021/acs.estlett.0c00798)

### Citation for code:

Ruyle, B. J.; Thackray, C. P.; McCord, J. P.; Strynar, M. J.; Mauge-Lewis, K. A.; Fenton, S. E.; Sunderland, E. M. Reconstructing the Composition of Per- and Polyfluoroalkyl Substances (PFAS) in Contemporary Aqueous Film Forming Foams. Environ. Sci. Technol. Lett. 2020. [https://doi.org/10.1021/acs.estlett.0c00798](http://dx.doi.org/10.1021/acs.estlett.0c00798).

# Command line options
usage: `infer_precursors.py [-h] [-d FILENAME] [-o OUTFILE_STEM] [-t TARGET] [-m MAX_STEPS] [-D MAX_DEPTH] istart [iend]`

Sample posterior for precursors.

positional arguments:  
&nbsp;&nbsp;&nbsp;&nbsp;`istart`&nbsp;&nbsp;&nbsp;&nbsp;`first sample index`  
&nbsp;&nbsp;&nbsp;&nbsp;`iend`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`last sample index`

optional arguments:
  `-h, --help`            show this help message and exit  
  `-d FILENAME, --datafile FILENAME`
                        location of measurements file  
  `-o OUTFILE_STEM, --outfile OUTFILE_STEM`
                        Stem for output filename. Will gain suffix N (sample index)  
  `-t TARGET, --target-steps TARGET`
                        Effective sample size to attain  
  `-m MAX_STEPS, --max-steps MAX_STEPS`
                        Maximum number of steps before quiting.  
  `-D MAX_DEPTH, --max-depth MAX_DEPTH`
                        Maximum depth of windowing in sampler tuning.  

# Input format
Measurement data should be contained in a csv with column headers for measured changes in concentrations of Cn PFCA, where n=number of per fluorinated carbons, after the TOP assay:  
`C3, C4, C5, C6, C7, C8, PFOS`  
(C3=PFBA, C4=PFPeA, C5=PFHxA, C6=PFHpA, C7=PFOA, C8=PFNA)  
and associated MDLs:  
`C3MDL, C4MDL, C5MDL, C6MDL, C7MDL, C8MDL, PFOSMDL`  
and associated measurement errors:  
`C3err, C4err, C5err, C6err, C7err, C8err, PFOSerr`  
as well as for a column for whether C8 (PFNA) was measured (True/False or 1/0):  
`C8incl`  
and the name of the prior to use:  
`prior_name`  

Method errors can be assessed by the relative difference of replicate analyses
of the same sample. You can use total method error by setting all err columns
equal to the total relative difference or compound specific method errors by
setting C3-C8err and PFOSerr to their own values.

# Output
The output samples are saved as binary format specified by Numpy (.npy)
and contain values of log10 of the precursor concentrations. Precursors in the
output file are index in python by column as:

`{0 : 4:2 FT precursors, 1 : 6:2 FT precursors, 2 : 8:2 FT precursors,
  3 : C4 ECF precursors, 4 : C5 ECF precursors, 5 : C6 ECF precursors
  6 : C7 ECF precursors, 7 : C8 ECF precursors}`

# Python dependencies
Python 3.X  
numpy  
pandas  
emcee  
