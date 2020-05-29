import numpy as np
from types import FunctionType
from warnings import warn
from scipy.special import gammaincinv
import scipy
import os.path
import os
import tempfile
import h5py

# Hamming window
def hammwin(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))


# print output based on verbosity level
def printer(string, level):
    global verbosity
    if verbosity >= level:
        print(string)


# parse spod arguments
def spod_parser(nt, nx, isrealx, window, weight, noverlap, conflvl, nmodes):

    # window size and type
    if window == "hamming" or window is None:
        nDFT = 2 ** np.floor(np.log2(nt / 10.0)).astype(int)
        window = hammwin(nDFT)
        window_name = "Hamming"

    elif np.size(window) == 1:
        nDFT = window
        window = hammwin(nDFT)
        window_name = "Hamming"

    else:
        window = window.flatten()
        nDFT = np.size(window)
        window_name = "user specified"

    # block overlap
    if not noverlap:
        noverlap = np.floor(0.5 * nDFT)
    elif noverlap > nDFT - 1:
        raise ValueError("Overlap too large")

    # inner product weight
    if weight is None:
        weight = np.ones(nx)
        weight_name = "uniform"

    elif not np.size(weight) == nx:
        raise ValueError(
            "Weights must have the same spatial dimensions as data.  weight: %d, nx: %d"
            % (np.size(weight), nx)
        )

    else:
        weight = weight.flatten()
        weight_name = "user_specified"

    # confidence interval
    if conflvl is not None:
        if conflvl is True:
            conflvl = 0.95
        elif (conflvl is not False) and not (conflvl > 0 and conflvl < 1):
            raise ValueError(
                "Confidence interval value must be either True (defaults to 0.95) or a decimal between 0 and 1."
            )

    # number of blocks
    nBlks = np.floor((nt - noverlap) / (nDFT - noverlap)).astype(int)

    # number of modes to save
    if nmodes is None:
        nmodes = nBlks
    else:
        assert type(nmodes) == int, "nmodes is integer"

    # test feasibility
    if nDFT < 4 or nBlks < 2:
        raise ValueError("Spectral estimation parameters not meaningful.")

    # display parameter summary
    printer("\nSPOD parameters\n------------------------------------", 1)

    if isrealx:
        printer("Spectrum type             : one-sided (real-valued signal)", 1)
    else:
        printer("Spectrum type             : two-sided (complex-valued signal)", 1)

    printer("No. of snaphots per block : %d" % nDFT, 1)
    printer("Block overlap             : %d" % noverlap, 1)
    printer("No. of blocks             : %d" % nBlks, 1)
    printer("Windowing fct. (time)     : %s" % window_name, 1)
    printer("Weighting fct. (space)    : %s" % weight_name, 1)

    return (window, weight, noverlap, nDFT, nBlks, conflvl, nmodes)


def spod(
    x,
    window="hamming",
    weight=None,
    noverlap=None,
    dt=1,
    mean=None,
    isreal=None,
    nt=None,
    conflvl=None,
    normvar=False,
    debug=0,
    lowmem=False,
    savefile=None,
    nmodes=None,
    savefreqs=None,
):
    """"
    Spectral proper orthogonal decomposition
    
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
    """
    # verbosity of output
    global verbosity
    verbosity = debug

    # In low memory mode, a save file must be specified
    if lowmem:
        assert savefile is not None, "savefile must be provided in lowmem mode"

    # Warn about overwriting file
    if savefile and os.path.exists(savefile):
        warn("%s will be overwritten" % savefile)

    # Get problem dimensions
    if isinstance(x, FunctionType) or callable(x):
        xfun = True
        if nt is None:
            warn(
                'Please specify number of snapshots in "opts.nt". Trying to use default value of 10000 snapshots.'
            )
            nt = 10000
        x0 = x(0)
        sizex = np.shape(x0)
        xtype = x0.dtype
        nx = np.prod(sizex)
        dim = [nt] + list(sizex)
    else:
        xfun = False
        dim = np.shape(x)
        nt = dim[0]
        nx = np.prod(dim[1:]).astype(int)

    # Determine if data is real or complex
    if isreal is not None:
        isrealx = isreal
    elif not xfun:
        isrealx = np.isrealobj(x)
    else:
        isrealx = np.isrealobj(x0)
        del x0

    # Fix datatype
    if not xfun:
        x = np.float64(x) if isrealx else np.complex128(x)

    # Parse parameters
    window, weight, noverlap, nDFT, nBlks, conflvl, nmodes = spod_parser(
        nt, nx, isrealx, window, weight, noverlap, conflvl, nmodes
    )

    # determine correction for FFT window gain
    winWeight = 1.0 / np.mean(window)

    # Use data mean if not provided through opts['mean']
    if type(mean) == str and mean == "blockwise":
        blk_mean = True
    else:
        blk_mean = False

    # Calculate mean
    if xfun:
        if (mean is not None) and (not blk_mean):
            x_mean = mean.flatten()
            mean_name = "user specified"
        elif blk_mean:
            mean_name = "blockwise mean"
        else:
            x_mean = 0
            warn(
                "No mean subtracted. Consider providing long-time mean through opts['mean'] for better accuracy at low frequencies."
            )
            mean_name = "0"
    else:
        if blk_mean:
            mean_name = "blockwise mean"
        else:
            x_mean = np.mean(x, axis=0).flatten()
            mean_name = "long-time (true) mean"

    printer("Mean                      : %s" % mean_name, 1)

    # obtain frequency axis
    if isrealx:
        f = np.arange(np.ceil(nDFT / 2) + 1) / dt / nDFT
    else:
        f = np.arange(nDFT) / dt / nDFT
        if np.mod(nDFT, 2) == 0:
            f[nDFT / 2 :] -= 1.0 / dt
        else:
            f[(nDFT + 1) / 2 :] -= 1.0 / dt

    nFreq = np.size(f)

    if savefreqs is None:
        savefreqs = np.arange(nFreq)

    # In low memory mode, open temporary file for Q_hat.  Else, hold in memory.
    if lowmem:
        tempf = tempfile.TemporaryFile()
        tempfh5 = h5py.File(tempf, "a")
        Q_hat = tempfh5.create_dataset(
            "Q_hat", (nFreq, nx, nBlks), dtype=np.cdouble, compression="gzip"
        )
    else:
        Q_hat = np.zeros((nFreq, nx, nBlks), dtype=np.cdouble)

    # loop over number of blocks and generate Fourier realizations
    printer("\nCalculating temporal DFT\n------------------------------------", 1)
    for iBlk in range(nBlks):
        offset = int(min(iBlk * (nDFT - noverlap) + nDFT, nt) - nDFT)
        timeIdx = np.arange(nDFT) + offset
        printer(
            "block %d  / %d (%d:%d)"
            % (iBlk + 1, nBlks, timeIdx[0] + 1, timeIdx[-1] + 1),
            2,
        )

        # build present block
        if blk_mean:
            x_mean = 0
        if xfun:
            Q_blk = np.zeros((nDFT, nx), dtype=xtype)
            for ti in timeIdx:
                Q_blk[ti - offset, :] = x(ti).flatten() - x_mean
        else:
            Q_blk = np.subtract(
                np.reshape(x[timeIdx, :], (nDFT, -1)), np.expand_dims(x_mean, 0)
            )

        # if block mean is to be subtracted, do it now that all data is collected
        if blk_mean:
            Q_blk = np.subtract(Q_blk, np.mean(Q_blk, axis=0, keepdims=True))

        # normalize by pointwise variance
        if normvar:
            Q_var = np.sum(
                np.power(np.subtract(Q_blk, np.mean(Q_blk, axis=0, keepdims=True)), 2),
                axis=0,
            ) / (nDFT - 1)
            # address division-by-0 problem with NaNs
            eps = np.finfo(Q_var.dtype).eps
            Q_var[Q_var < 4 * eps] = 1
            Q_blk = np.divide(Q_blk, Q_var)

        # window and Fourier transform block
        Q_blk = np.multiply(Q_blk, np.expand_dims(window, axis=1))
        Q_blk_hat = (winWeight / nDFT) * scipy.fft.fft(Q_blk, axis=0, workers=-1)
        Q_blk_hat = Q_blk_hat[:nFreq, :]

        # correct Fourier coefficients for one-sided spectrum
        if isrealx:
            Q_blk_hat[1:-1, :] *= 2.0

        # keep FFT blocks in memory
        Q_hat[:, :, iBlk] = Q_blk_hat

    # Dimensions of P
    pDim = [nFreq] + list(dim[1:]) + [nmodes]

    # In low memory mode, save L and P to files.  Else, hold in memory.
    if lowmem:
        output = h5py.File(savefile, "a")
        if "L" in output:
            del output["L"]
        L = output.create_dataset(
            "L", (nFreq, nBlks), dtype=np.double, compression="gzip"
        )
        if nmodes > 0:
            if "P" in output:
                del output["P"]
            P = output.create_dataset("P", pDim, dtype=np.cdouble, compression="gzip")
    else:
        L = np.zeros((nFreq, nBlks), dtype=np.double)
        if nmodes > 0:
            P = np.zeros(pDim, dtype=np.cdouble)

    # loop over all frequencies and calculate SPOD
    printer("\nCalculating SPOD\n------------------------------------", 1)
    for iFreq in savefreqs:
        printer("frequency %d / %d (f=%g)" % (iFreq + 1, nFreq, f[iFreq]), 2)
        Q_hat_f = Q_hat[iFreq, :, :]
        M = (
            np.matmul(
                np.conj(np.transpose(Q_hat_f)),
                np.multiply(Q_hat_f, np.expand_dims(weight, axis=1)),
            )
            / nBlks
        )
        Lambda, Theta = scipy.linalg.eig(
            M
        )  # Lambda matches but Theta does not (but is still valid)
        idx = np.argsort(Lambda)[::-1]
        Lambda = Lambda[idx]
        if nmodes > 0:
            Theta = Theta[:, idx]
            Psi = np.matmul(
                np.matmul(Q_hat_f, Theta),
                np.diag(np.reciprocal(np.lib.scimath.sqrt(Lambda)) / np.sqrt(nBlks)),
            )
            P[iFreq, :] = Psi[:, :nmodes].reshape(pDim[1:])  # mode
        L[iFreq, :] = np.abs(Lambda)  # energy distribution

    # Calculate confidence interval
    if conflvl:
        if lowmem:
            Lc = output.create_dataset(
                "Lc", shape=(nFreq, nBlks, 2), dtype=L.dtype, compression="gzip"
            )
        else:
            Lc = np.zeros((nFreq, nBlks, 2), dtype=L.dtype)
        Lc[:, :, 0] = L * nBlks / gammaincinv(nBlks, conflvl)
        Lc[:, :, 1] = L * nBlks / gammaincinv(nBlks, 1.0 - conflvl)

    # If held in memory, create output dictionary
    if not lowmem:
        output = {"L": L, "f": f}
        if nmodes > 0:
            output["P"] = P
        if conflvl:
            output["Lc"] = Lc

    # Create output save file
    if savefile:
        if lowmem:
            tempfh5.close()
            tempf.close()
            if "f" in output:
                del output["f"]
            output.create_dataset("f", data=f, compression="gzip")
        else:
            with h5py.File(savefile, "a") as outputfile:
                for key, value in output.items():
                    if key in outputfile:
                        del outputfile[key]
                    outputfile.create_dataset(key, data=value, compression="gzip")

    # Return either in-memory or on-disk dictionary
    return output
