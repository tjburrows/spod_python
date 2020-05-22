# -*- coding: utf-8 -*-
import numpy as np
from types import FunctionType
from warnings import warn
from scipy.special import gammaincinv
import scipy

def hammwin(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

def nextpow2(n):
    return np.ceil(np.log2(np.abs(n)))

def printer(string, level):
    global verbosity
    if verbosity >= level:
        print(string)

def spod_parser(nt, nx, isrealx, window, weight, noverlap):
    
    if window == 'hamming' or not window:
        nDFT = 2 ** np.floor(np.log2(nt / 10.0)).astype(int)
        window = hammwin(nDFT)
        window_name = 'Hamming'

    elif np.size(window) == 1:
        nDFT = window;
        window = hammwin(nDFT)
        window_name = 'Hamming'

    else:
        nDFT = np.size(window)
        window_name = 'user specified'
    
    # block overlap
    if not noverlap:
        noverlap = np.floor(0.5 * nDFT)
    elif noverlap > nDFT - 1:
        raise ValueError('Overlap too large')
    
    # inner product weight
    if not weight:
        weight = np.ones(nx)
        weight_name = 'uniform'
        
    elif not np.size(weight) == nx:
        raise ValueError('Weights must have the same spatial dimensions as data.')
        
    else:
        weight_name = 'user_specified'
    
    # number of blocks
    nBlks = np.floor((nt - noverlap) / (nDFT - noverlap)).astype(int)
    
    # test feasibility
    if nDFT < 4 or nBlks < 2:
        raise ValueError('Spectral estimation parameters not meaningful.')
    
    # display parameter summary
    printer('\nSPOD parameters\n------------------------------------', 1)
    
    if isrealx:
        printer('Spectrum type             : one-sided (real-valued signal)', 1)
    else:
        printer('Spectrum type             : two-sided (complex-valued signal)', 1)
    
    printer('No. of snaphots per block : %d' % nDFT, 1)
    printer('Block overlap             : %d' % noverlap, 1)
    printer('No. of blocks             : %d' % nBlks, 1)
    printer('Windowing fct. (time)     : %s' % window_name, 1)
    printer('Weighting fct. (space)    : %s' % weight_name, 1)
    
    return (window, weight, noverlap, nDFT, nBlks)

def spod(x, window='hamming', weight=None, noverlap=None, dt=1, mean=None, isreal=None, nt = None, confint=False, conflvl=0.95, normvar=False, debug=0):
    global verbosity
    verbosity = debug
    
    # Get problem dimensions
    if isinstance(x, FunctionType) or callable(x):
        xfun = True
        if nt is None:
            warn('Please specify number of snapshots in "opts.nt". Trying to use default value of 10000 snapshots.')
            nt  = 10000
        sizex = np.shape(x)[0]
        nx = np.size(x)
        dim = [nt, sizex]
    else:
        xfun = False
        dim = np.shape(x)
        nt = dim[0]
        nx = np.prod(dim[1:]).astype(int)

    # Determine whether data is real-valued or complex-valued-valued to decide on one- or two-sided
    # spectrum. If opts['isreal'] is not set, determine from data. If data is
    # provided through a function handle XFUN and opts.isreal is not specified,
    # determine complex-valuedity from first snapshot.
    if isreal is not None:
        isrealx = isreal
    elif not xfun:
        isrealx = np.isrealobj(x)
    else:
        isrealx = np.isrealobj(x[0])
    
    window,weight,noverlap,nDFT,nBlks = spod_parser(nt, nx, isrealx, window, weight, noverlap)
    
    # determine correction for FFT window gain
    winWeight   = 1.0 / np.mean(window)
    
    # Use data mean if not provided through opts['mean']
    if mean == 'blockwise':
        blk_mean = True
    else:
        blk_mean = False
    
    if xfun:
        if (mean is not None) and (not blk_mean):
            x_mean = mean
            mean_name = 'user specified'
        elif blk_mean:
            mean_name = 'blockwise mean'
        else:
            x_mean = 0
            warn('No mean subtracted. Consider providing long-time mean through opts[\'mean\'] for better accuracy at low frequencies.')
            mean_name   = '0'
    else:
        if blk_mean:
            mean_name = 'blockwise mean'
        else:
            x_mean = np.mean(x, axis=0).flatten(order='F')
            mean_name = 'long-time (true) mean'
    
    printer('Mean                      : %s' % mean_name, 1)
    
    # obtain frequency axis
    if isrealx:
        f = np.arange(np.ceil(nDFT / 2) + 1) / dt / nDFT
    else:
        f = np.arange(nDFT) / dt / nDFT
        if np.mod(nDFT, 2) == 0:
            f[nDFT/2:] -= 1.0 / dt
        else:
            f[(nDFT+1)/2:] -= 1.0 / dt
    nFreq = np.size(f)
    
    # loop over number of blocks and generate Fourier realizations
    printer('\nCalculating temporal DFT\n------------------------------------', 1)
    Q_hat = np.zeros((nFreq, nx, nBlks), dtype=np.cdouble)
    for iBlk in range(nBlks):
        offset = int(min(iBlk * (nDFT - noverlap) + nDFT, nt) - nDFT)
        timeIdx = np.arange(nDFT) + offset
        printer('block %d  / %d (%d:%d)' % (iBlk+1, nBlks, timeIdx[0]+1, timeIdx[-1]+1), 2)
        # build present block
        if blk_mean:
            x_mean = 0
        if xfun:
            Q_blk = np.zeros((nDFT, nx))
            for ti in timeIdx:
                xi = x(ti)
                Q_blk[ti - offset, :] = xi.flatten(order='F') - x_mean
        else:
            Q_blk = np.subtract(np.reshape(x[timeIdx,:], (nDFT,-1), order='F'), np.expand_dims(x_mean,0))
        
        # if block mean is to be subtracted, do it now that all data is collected
        if blk_mean:
            Q_blk = np.subtract(Q_blk, np.mean(Q_blk, axis=0, keepdims=True))
        
        # normalize by pointwise variance
        if normvar:
            Q_var = np.sum(np.power(np.minus(Q_blk, np.mean(Q_blk, axis=0, keepdims=True)), 2), axis=0) / (nDFT - 1)
            # address division-by-0 problem with NaNs
            eps = np.finfo(type(Q_var)).eps
            Q_var[Q_var < 4 * eps] = 1
            Q_blk = np.divide(Q_blk,Q_var)
            
        # window and Fourier transform block
        Q_blk = np.multiply(Q_blk, np.expand_dims(window, axis=1))
        Q_blk_hat = (winWeight / nDFT) * scipy.fft.fft(Q_blk, axis=0, workers=-1)[:nFreq, :]
        
    #     # correct Fourier coefficients for one-sided spectrum
        if isrealx:
            Q_blk_hat[1:-1, :] *= 2.0
        
    #     # keep FFT blocks in memory
        Q_hat[:,:,iBlk] = Q_blk_hat
    
    # loop over all frequencies and calculate SPOD
    L = np.zeros((nFreq, nBlks),  dtype=np.cdouble)
    printer('\nCalculating SPOD\n------------------------------------', 1)
   
    # keep everything in memory
    P   = np.zeros((nFreq,nx,nBlks), dtype=np.cdouble)
    for iFreq in range(nFreq):
        printer('frequency %d / %d (f=%g)' % (iFreq+1, nFreq, f[iFreq]), 2)
        Q_hat_f = np.matrix(Q_hat[iFreq, :, :])
        M = np.matmul(Q_hat_f.H, np.multiply(Q_hat_f, np.expand_dims(weight, axis=1))) / nBlks
        Lambda, Theta = scipy.linalg.eig(M,check_finite=True) # Lambda matches but Theta does not (but is still valid)
        Lambda = np.real_if_close(Lambda)
        Theta = np.real_if_close(Theta)
        idx = np.argsort(Lambda)[::-1]
        Lambda = Lambda[idx]
        Theta = Theta[:, idx]
        Psi = np.matmul(np.matmul(Q_hat_f, Theta), np.diag(np.reciprocal(np.sqrt(Lambda)) / np.sqrt(nBlks)))
        P[iFreq,:,:] = Psi # mode
        L[iFreq, :] = np.abs(Lambda) # energy distribution
        if confint:
            xi2_upper = 2 * gammaincinv(nBlks, 1.0 - conflvl)
            xi2_lower = 2 * gammaincinv(nBlks, conflvl)
            Lc = np.zeros((nFreq, nBlks, 2))
            Lc[iFreq, :, 0] = L[iFreq, :] * 2 * nBlks / xi2_lower
            Lc[iFreq, :, 1] = L[iFreq, :] * 2 * nBlks / xi2_upper
    
    newDim = [nFreq]
    newDim.extend(dim[1:])
    newDim.append(nBlks)
    P = np.reshape(P, tuple(newDim), order='F')
    L = np.real_if_close(L)
    P = np.real_if_close(P)
    output = {'L':L, 'P':P, 'f':f}
    if confint:
        output['Lc'] = Lc
    return output