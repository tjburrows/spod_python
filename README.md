# Spectral Proper Orthogonal Decomposition in Python
This is a port of the [Towne et al. SPOD Matlab function](https://github.com/SpectralPOD/spod_matlab) to Python 3.

## Features
- Calculate SPOD on N-dimensional data, where time is first dimension
- Same set of SPOD options as [original Matlab version](https://github.com/SpectralPOD/spod_matlab) (window,weight,dt,noverlap,normvar)
- Keep data in memory or use little memory and keep on disk (low memory mode)
- Can save results to disk (HDF5 file), in addition to or instead of returning in-memory results

## Requirements
 Install the below requirements with `pip install requirements.txt`
 - spod: numpy, scipy, h5py for low-memory mode
 - examples: matplotlib, h5py

## Files
| File        |     Description     |
|:-------------|:-------------|
| spod.py | Spectral proper orthogonal decomposition in Matlab | 
| example_1.py | Inspect data and plot SPOD spectrum | 
| example_2.py | Plot SPOD spectrum and inspect SPOD modes | 
| example_3.py | Specify spectral estimation parameters and use weighted inner product | 
| example_4.py | Calculate the SPOD of large data and save results on hard drive | 
| example_5.py | Calculate full SPOD spectrum of large data | 
| example_6.py | Calculate and plot confidence intervals for SPOD eigenvalues | 
| utils.py/getjet | Interfaces external data source with SPOD() (examples 4-5) | 
| utils.py/trapzWeightsPolar | Integration weight matrix for cylindrical coordinates (examples 3-6) | 
| jet_data/jetLES.mat | Mach 0.9 turbulent jet test database | 

## Usage
    spod(x, window='hamming', weight=None, noverlap=None, dt=1, mean=None, isreal=None, nt=None, conflvl=None, normvar=False,  debug=0, lowmem=False, savefile=None, nmodes=None, savefreqs=None)
    
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
        a file handle. Default is False, keeping everything in memory.  If
        lowmem is True, savefile must be specified for the returned data.  This mode requires
        the h5py package.
    savefile : string, optional
        Filename to which to save the results in HDF5 format.  If lowmem is True,
        a handle for this file is returned.  If False or None, the in-memory results 
        are returned in a dictionary.  If file exists, it is overwritten.  Defaults to None/False.
    nmodes : int, optional
        Number of most energetic SPOD modes to be saved.  Defaults to all modes.
    savefreqs: list of ints, optional
        List of frequency indices to calculate modes (P) and spectral energies (L).  Meant 
        to reduce size of data if not all frequences are needed.  Defaults to all frequences.
    
    Returns
    -------
    output
        A dictionary containing SPOD modes (P), modal energy spectra (L), and frequency
        vector (f).  If conflvl is specified, confidence interval is returned (Lc).  If
        lowmem is True, a file object is returned which points to the on-disk HDF5 dataset.



## References
[1] Towne, A., Schmidt, O. T., Colonius, T., *Spectral proper orthogonal decomposition and its relationship to dynamic mode decomposition and resolvent analysis*, [arXiv:1708.04393](https://arxiv.org/abs/1708.04393), 2017

[2] Lumley, J. L., *Stochastic tools in turbulence*, Academic Press, 1970

[3] G. A. Brès, P. Jordan, M. Le Rallic, V. Jaunet, A. V. G. Cavalieri, A. Towne, S. K. Lele, T. Colonius, O. T. Schmidt,  *Importance of the nozzle-exit boundary-layer state in subsonic turbulent jets*, submitted to JFM, 2017

[4] Schmidt, O. T., Towne, A., Rigas, G.,  Colonius, T., Bres, G. A., *Spectral analysis of jet turbulence*, [arXiv:1711.06296](https://arxiv.org/abs/1711.06296), 2017    
