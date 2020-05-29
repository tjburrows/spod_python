# Spectral Proper Orthogonal Decomposition in Python
This is a port of the Towne et al. SPOD Matlab function to Python 3.

## Current Status
Basic spod computation works, and example results match well.  Results do not match exactly in part because of different (but still valid) eigenvectors from Numpy.

### To Do
 - commenting and formatting

### Requirements
 Install the below requirements with `pip install requirements.txt`
 - spod: numpy, scipy, h5py for low-memory mode
 - examples: matplotlib, h5py

### Usage
    spod(x, window='hamming', weight=None, noverlap=None, dt=1, mean=None, isreal=None, nt=None, conflvl=None, normvar=False,  debug=0, lowmem=False, savefile=None)
    
    Parameters
    ----------
    x : array or function object
        Data array whose first dimension is time, or function that retrieves
        one snapshot at a time like x(i).  x(i), like x, can have any dimension. 
        If x is a function, it is recommended to specify the total number of 
        snaphots in nt (see below). If not specified, nt defaults to 10000. 
        isreal should be specified if a two-sided spectrum is desired even 
        though the data is real-valued, or if the data is initially real-valued,
        but complex-valued-valued for later snaphots.
    window : vector, int, or string, optional
        A temporal window. If WINDOW is a vector, X
        is divided into segments of the same length as WINDOW. Each segment is
        then weighted (pointwise multiplied) by WINDOW. If WINDOW is a scalar,
        a Hamming window of length WINDOW is used. If WINDOW is none or 'hamming',
        a Hamming window is used.
    weight : array, optional
        A spatial inner product weight.  SPOD modes are optimally ranked and 
        orthogonal at each frequency. WEIGHT must have the same spatial 
        dimensions as x.
    noverlap : int, optional
        Number of snaptions to overlap consecutive blocks.  noverlap defaults
        to 50% of the length of WINDOW if not specified.
    dt : float, optional
        Time step between consecutive snapshots to determine a physical 
        frequency F.  dt defaults to 1 if not specified.
    mean : array or string, optional
        A mean that is subtracted from each snapshot.  If 'blockwise', the mean
        of each block is subtracted from itself.  If x is a function the mean
        provided is a temporal mean.
    isreal : bool, optional
        Describes if x data is real.
    nt : int, optional
        Number of snapshots.  If x is an array, this is determined from x
        dimensions.  If x is a function, this defaults to 10000 if not specified.
    conflvl : bool or float, optional
        Calculate and return confidence interval levels of L (Lc).  If True,
        the lower and upper 95% confidence levels of the j-th
        most energetic SPOD mode at the i-th frequency are returned in
        Lc(i,j,1) and Lc(i,j,2), respectively.  If a float between 0 and 1, 
        the conflvl*100% confidence interval of L is returned. A 
        chi-squared distribution is used, i.e. we assume a standard normal 
        distribution of the SPOD eigenvalues.  Defaults to None/False.
    normvar : bool, optional
        Normalize each block by pointwise variance.  Defaults to false.
    debug : {0, 1, 2}, optional
        Verbosity of output.  0 hides output.  1 shows some output, 2 shows all output.
        Defaults to 0.
    lowmem : bool, optional
        Specifies whether to use low-memory mode.  If True, this stores the FFT blocks in a
        temporary file on disk, and also stores all returned quantities on disk, returning
        a h5py handle for the file. Default is False, keeping everything in memory.  If
        lowmem is True, savefile must be specified for the returned data.
    savefile : string, optional
        Filename to which to save the results in HDF5 format.  If lowmem is True,
        a handle for this file is returned.  If False or None, the in-memory results 
        are returned in a dictionary.  Defaults to None/False.
    
    Returns
    -------
    output
        A dictionary containing SPOD modes (P), modal energy spectra (L), and frequency
        vector (f).  If conflvl is specified, confidence interval is returned (Lc).  If
        lowmem is True, a file object is returned which points to the on-disk HDF5 dataset.

## Original Matlab Documentation
SPOD() is a Matlab implementation of the frequency domain form of proper orthogonal decomposition (POD, also known as principle component analysis or Karhunen-Loève decomposition) called spectral proper orthogonal decomposition (SPOD). SPOD is derived from a space-time POD problem for stationary flows [[1](https://arxiv.org/abs/1708.04393),2] and leads to modes that each oscillate at a single frequency. SPOD modes represent dynamic structures that optimally account for the statistical variability of stationary random processes.

The large-eddy simulation data provided along with this example is a subset of the database of a Mach 0.9 turbulent jet described in [3] and was calculated using the unstructured flow solver Charles developed at Cascade Technologies. If you are using the database in your research or teaching, please include explicit mention of Brès et al. [3]. The test database consists of 5000 snapshots of the symmetric component (m=0) of a round turbulent jet. 

`spod.m` is a stand-alone Matlab function with no toolbox dependencies. All other Matlab files contained in this repository are related to the six examples that demonstrate the functionality of the code (see file descriptions below). A physical interpretation of the results obtained from the examples can be found in [[4](https://arxiv.org/abs/1711.06296)].

## Download

### Using your browser

Repository zip file with examples (81.5 MB): [https://github.com/SpectralPOD/spod_matlab/archive/master.zip](https://github.com/SpectralPOD/spod_matlab/archive/master.zip)

Matlab function only (15 KB): [https://raw.githubusercontent.com/SpectralPOD/spod_matlab/master/spod.m](https://raw.githubusercontent.com/SpectralPOD/spod_matlab/master/spod.m)

### Using Git in the terminal
git clone https://github.com/SpectralPOD/spod_matlab.git

## Files
| File        |     Description     |
| ------------- |:-------------|
| spod.m | Spectral proper orthogonal decomposition in Matlab | 
| example_1.m | Inspect data and plot SPOD spectrum | 
| example_2.m | Plot SPOD spectrum and inspect SPOD modes | 
| example_3.m | Specify spectral estimation parameters and use weighted inner product | 
| example_4.m | Calculate the SPOD of large data and save results on hard drive | 
| example_5.m | Calculate full SPOD spectrum of large data | 
| example_6.m | Calculate and plot confidence intervals for SPOD eigenvalues | 
| jet_data/getjet.m | Interfaces external data source with SPOD() (examples 4-5) | 
| utils/trapzWeightsPolar.m | Integration weight matrix for cylindrical coordinates (examples 3-6) | 
| utils/jetLES.mat | Mach 0.9 turbulent jet test database | 
| LICENSE.txt | License | 

## Usage
  [L,P,F] = SPOD(X) returns the spectral proper orthogonal decomposition
  of the data matrix X whose first dimension is time. X can have any
  number of additional spatial dimensions or variable indices.
  The columns of L contain the modal energy spectra. P contains the SPOD
  modes whose spatial dimensions are identical to those  
  of X. The first index of P is the frequency and
  the last one the mode number ranked in descending order 
  by modal energy. F is the frequency vector. If DT is not specified, a
  unit frequency sampling is assumed. For real-valued data, one-sided spectra
  are returned. Although SPOD(X) automatically chooses default spectral 
  estimation parameters, it is strongly encouraged to manually specify
  problem-dependent parameters on a case-to-case basis.

  [L,P,F] = SPOD(X,WINDOW) uses a temporal window. If WINDOW is a vector, X
  is divided into segments of the same length as WINDOW. Each segment is
  then weighted (pointwise multiplied) by WINDOW. If WINDOW is a scalar,
  a Hamming window of length WINDOW is used. If WINDOW is omitted or set
  as empty, a Hamming window is used.

  [L,P,F] = SPOD(X,WINDOW,WEIGHT) uses a spatial inner product weight in
  which the SPOD modes are optimally ranked and orthogonal at each
  frequency. WEIGHT must have the same spatial dimensions as X. 

  [L,P,F] = SPOD(X,WINDOW,WEIGHT,NOVERLAP) increases the number of
  segments by overlapping consecutive blocks by NOVERLAP snapshots.
  NOVERLAP defaults to 50% of the length of WINDOW if not specified. 

  [L,P,F] = SPOD(X,WINDOW,WEIGHT,NOVERLAP,DT) uses the time step DT
  between consecutive snapshots to determine a physical frequency F. 

  [L,P,F] = SPOD(XFUN,...,OPTS) accepts a function handle XFUN that 
  provides the i-th snapshot as x(i) = XFUN(i). Like the data matrix X, x(i) can 
  have any dimension. It is recommended to specify the total number of 
  snaphots in OPTS.nt (see below). If not specified, OPTS.nt defaults to 10000.
  OPTS.isreal should be specified if a two-sided spectrum is desired even
  though the data is real-valued, or if the data is initially real-valued, 
  but complex-valued-valued for later snaphots.

  [L,P,F] = SPOD(X,WINDOW,WEIGHT,NOVERLAP,DT,OPTS) specifies options:
  OPTS.savefft: save FFT blocks to avoid storing all data in memory [{false} | true]
  OPTS.deletefft: delete FFT blocks after calculation is completed [{true} | false]
  OPTS.savedir: directory where FFT blocks and results are saved [ string | {'results'}]
  OPTS.savefreqs: save results for specified frequencies only [ vector | {all} ]
  OPTS.mean: provide a mean that is subtracted from each snapshot [ array of size X | {temporal mean of X; 0 if XFUN} ]
  OPTS.nsave: number of most energtic modes to be saved [ integer | {all} ]
  OPTS.isreal: complex-valuedity of X or represented by XFUN [{determined from X or first snapshot if XFUN is used} | logical ]
  OPTS.nt: number of snapshots [ integer | {determined from X; defaults to 10000 if XFUN is used}]
  OPTS.conflvl: confidence interval level [ scalar between 0 and 1 | {0.95} ]

  [L,PFUN,F] = SPOD(...,OPTS) returns a function PFUN instead of the SPOD
  data matrix P if OPTS.savefft is true. The function returns the j-th
  most energetic SPOD mode at the i-th frequency as p = PFUN(i,j) by
  reading the modes from the saved files. Saving the data on the hard
  drive avoids memory problems when P is large.

  [L,P,F,Lc] = SPOD(...) returns the confidence interval Lc of L. By
  default, the lower and upper 95% confidence levels of the j-th
  most energetic SPOD mode at the i-th frequency are returned in
  Lc(i,j,1) and Lc(i,j,2), respectively. The OPTS.conflvl*100% confidence
  interval is returned if OPTS.conflvl is set. For example, by setting 
  OPTS.conflvl = 0.99 we obtain the 99% confidence interval. A 
  chi-squared distribution is used, i.e. we assume a standard normal 
  distribution of the SPOD eigenvalues.

## References
[1] Towne, A., Schmidt, O. T., Colonius, T., *Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis*, [arXiv:1708.04393](https://arxiv.org/abs/1708.04393), 2017

[2] Lumley, J. L., *Stochastic tools in turbulence*, Academic Press, 1970

[3] G. A. Brès, P. Jordan, M. Le Rallic, V. Jaunet, A. V. G. Cavalieri, A. Towne, S. K. Lele, T. Colonius, O. T. Schmidt,  *Importance of the nozzle-exit boundary-layer state in subsonic turbulent jets*, submitted to JFM, 2017

[4] Schmidt, O. T., Towne, A., Rigas, G.,  Colonius, T., Bres, G. A., *Spectral analysis of jet turbulence*, [arXiv:1711.06296](https://arxiv.org/abs/1711.06296), 2017    
