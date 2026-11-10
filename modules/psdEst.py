import numpy as np
import scipy.io as sio
from scipy.io import savemat,loadmat
import scipy.signal as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import math as mt
import os.path as op
import obspy
import statistics as st
from obspy.core import read
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy import UTCDateTime
from obspy import read_inventory
from obspy.clients.fdsn.header import FDSNNoDataException
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import cdist
from scipy.signal import periodogram
from scipy.ndimage import uniform_filter1d, median_filter,gaussian_filter1d
import datetime
from obspy.clients.filesystem.sds import Client

def computeHourlyPSDs(processedAll,hourStart,psdLen=120.0,psdHop=60.0,extractLen=3600.0,
                      overlapBuffer=60.0,detrend="constant",window="hann"):
    """
    Compute one PSD for every 120-second window.

    The processed traces are expected to begin `overlapBuffer` seconds
    before hourStart and continue through the end of the analysis hour.

    Parameters
    ----------
    processedAll : list of obspy.Stream
        One processed Stream per requested channel. Empty Streams are
        retained for missing channels.
    hourStart : obspy.UTCDateTime
        Beginning of the actual analysis hour.
    psdLen : float
        PSD segment length in seconds.
    psdHop : float
        Time shift between consecutive PSD segments in seconds.
    extractLen : float
        Length of the actual analysis hour.
    overlapBuffer : float
        Data retained before hourStart.
    detrend : str or callable
        Detrending passed to scipy.signal.periodogram.
    window : str or tuple
        Window passed to scipy.signal.periodogram.

    Returns
    -------
    psdAll : list
        List with the same length as processedAll.

        For an available channel, each entry is a dictionary containing:

            frequencies : array, shape (nFrequency,)
            psd         : array, shape (60, nFrequency)
            window_start_times : list of UTCDateTime
            sampling_rate
            seed_id

        For an unavailable channel, the entry is None.
    """

    totalDuration = overlapBuffer + extractLen

    nWindowsExpected = int(np.floor((totalDuration - psdLen) / psdHop)) + 1

    psdAll = []
    psdAllAvg = []

    for streamNo, st in enumerate(processedAll):

        if st is None or len(st) == 0:
            psdAll.append(None)
            continue

        if len(st) != 1:
            raise ValueError(f"processedAll[{streamNo}] contains {len(st)} traces. ""Expected one merged trace.")

        tr = st[0]

        samplingRate = float(tr.stats.sampling_rate)

        nSamplesWindow = int(round(psdLen * samplingRate))
        nSamplesHop = int(round(psdHop * samplingRate))
        nSamplesExpected = int(round(totalDuration * samplingRate))

        data = np.asarray(tr.data, dtype=np.float64)

        # ObsPy may include the endpoint, producing one extra sample.
        # Retain exactly 3660 * samplingRate samples.
        if data.size < nSamplesExpected:
            print(f"{tr.id}: expected {nSamplesExpected} samples, "f"but found only {data.size}. Skipping this channel.")
            psdAll.append(None)
            continue

        data = data[:nSamplesExpected]

        psdList = []
        windowStartTimes = []

        frequencies = None

        for windowNo in range(nWindowsExpected):

            startIndex = windowNo * nSamplesHop
            endIndex = startIndex + nSamplesWindow

            segment = data[startIndex:endIndex]

            if segment.size != nSamplesWindow:
                raise RuntimeError(
                    f"{tr.id}, window {windowNo}: expected "
                    f"{nSamplesWindow} samples but obtained "
                    f"{segment.size}."
                )

            frequenciesNow, psdNow = periodogram(segment,fs=samplingRate,window=window,detrend=detrend,
                                                 return_onesided=True,scaling="density")

            if frequencies is None:
                frequencies = frequenciesNow

            psdList.append(psdNow)

            windowStartTimes.append(tr.stats.starttime + startIndex / samplingRate)

        psdMatrix = np.asarray(psdList, dtype=np.float64)
        psdMatrixAvg = psdMatrix.mean(axis=0)

        psdAll.append({
            "seed_id": tr.id,
            "sampling_rate": samplingRate,
            "frequencies": frequencies,
            "psd": psdMatrix,
            "window_start_times": windowStartTimes,
        })
        
        psdAllAvg.append({
            "seed_id": tr.id,
            "sampling_rate": samplingRate,
            "frequencies": frequencies,
            "psd": psdMatrixAvg[None,:],
            "window_start_times": windowStartTimes,
        })

    return psdAll, psdAllAvg

def computeHourlyDeGlitchPSDs(processedAll,hourStart,glitchInt, psdLen=120.0,psdHop=60.0,extractLen=3600.0,
                      overlapBuffer=60.0,detrend="constant",window="hann"):
    """
    Compute one PSD for every 120-second window.

    The processed traces are expected to begin `overlapBuffer` seconds
    before hourStart and continue through the end of the analysis hour.

    Parameters
    ----------
    processedAll : list of obspy.Stream
        One processed Stream per requested channel. Empty Streams are
        retained for missing channels.
    hourStart : obspy.UTCDateTime
        Beginning of the actual analysis hour.
    psdLen : float
        PSD segment length in seconds.
    psdHop : float
        Time shift between consecutive PSD segments in seconds.
    extractLen : float
        Length of the actual analysis hour.
    overlapBuffer : float
        Data retained before hourStart.
    detrend : str or callable
        Detrending passed to scipy.signal.periodogram.
    window : str or tuple
        Window passed to scipy.signal.periodogram.

    Returns
    -------
    psdAll : list
        List with the same length as processedAll.

        For an available channel, each entry is a dictionary containing:

            frequencies : array, shape (nFrequency,)
            psd         : array, shape (60, nFrequency)
            window_start_times : list of UTCDateTime
            sampling_rate
            seed_id

        For an unavailable channel, the entry is None.
    """

    totalDuration = overlapBuffer + extractLen

    nWindowsExpected = int(np.floor((totalDuration - psdLen) / psdHop)) + 1

    psdAll = []
    psdAllAvg = []
    
    for streamNo, st in enumerate(processedAll):

        if st is None or len(st) == 0:
            psdAll.append(None)
            continue

        if len(st) != 1:
            raise ValueError(f"processedAll[{streamNo}] contains {len(st)} traces. ""Expected one merged trace.")

        tr = st[0]

        samplingRate = float(tr.stats.sampling_rate)

        nSamplesWindow = int(round(psdLen * samplingRate))
        nSamplesHop = int(round(psdHop * samplingRate))
        nSamplesExpected = int(round(totalDuration * samplingRate))

        data = np.asarray(tr.data, dtype=np.float64)

        # ObsPy may include the endpoint, producing one extra sample.
        # Retain exactly 3660 * samplingRate samples.
        if data.size < nSamplesExpected:
            print(f"{tr.id}: expected {nSamplesExpected} samples, "f"but found only {data.size}. Skipping this channel.")
            psdAll.append(None)
            continue

        data = data[:nSamplesExpected]

        psdList = []
        windowStartTimes = []

        frequencies = None

        for windowNo in range(nWindowsExpected):
            
            startIndex = windowNo * nSamplesHop
            endIndex = startIndex + nSamplesWindow
            if(glitchInt[streamNo] is not None):
                #foundSInd = any(low <= startIndex <= high for low, high in glitchInt[streamNo])
                #foundEInd = any(low <= endIndex <= high for low, high in glitchInt[streamNo])
                hasGlitch = any(startIndex <= high and endIndex > low
                                for low, high in glitchInt[streamNo])
            else:
                hasGlitch = False
            
            if(not hasGlitch):
        
                segment = data[startIndex:endIndex]

                if segment.size != nSamplesWindow:
                    raise RuntimeError(
                        f"{tr.id}, window {windowNo}: expected "
                        f"{nSamplesWindow} samples but obtained "
                        f"{segment.size}."
                    )

                frequenciesNow, psdNow = periodogram(segment,fs=samplingRate,window=window,detrend=detrend,
                                                 return_onesided=True,scaling="density")

                if frequencies is None:
                    frequencies = frequenciesNow

                psdList.append(psdNow)

                windowStartTimes.append(tr.stats.starttime + startIndex / samplingRate)

        psdMatrix = np.asarray(psdList, dtype=np.float64)
        
        # get the average and return that as a different dictionary
        psdMatrixAvg = psdMatrix.mean(axis=0)

        psdAll.append({
            "seed_id": tr.id,
            "sampling_rate": samplingRate,
            "frequencies": frequencies,
            "psd": psdMatrix,
            "window_start_times": windowStartTimes,
        })
        
        psdAllAvg.append({
            "seed_id": tr.id,
            "sampling_rate": samplingRate,
            "frequencies": frequencies,
            "psd": psdMatrixAvg[None,:],
            "window_start_times": windowStartTimes,
        })
        

    return psdAll, psdAllAvg

def computeHourlyPSDMB(processedAll,hourStart,glitchInt,extractLen=3600.0,overlapBuffer=60.0,nSegments=13,
                       overlap=0.75,detrend="constant",window="hann"):
    """
    Compute McNamara-Buland-style hourly PSDs.

    The nominal analysis hour is divided into 13 overlapping
    subsegments with 75% overlap. For a 3600-s hour this gives
    900-s segments with a 225-s hop.

    The subsegment PSDs are computed only once and used to form
    two hourly PSD products:

      1. psdAllMB
         Mean PSD of all 13 subsegments.

      2. psdAllMBDeGlitch
         Mean PSD of only those subsegments that do not overlap
         any supplied glitch interval.

    If no clean subsegments remain for a channel, the corresponding
    entry in psdAllMBDeGlitch is None.

    Parameters
    ----------
    processedAll : list of obspy.Stream
        One processed Stream per channel. Each processed trace is
        expected to begin overlapBuffer seconds before hourStart
        and extend through the nominal 1-hour analysis interval.

    hourStart : obspy.UTCDateTime
        Beginning of the nominal analysis hour.

    glitchInt : list
        One entry per channel. Each entry is either None or a list
        of (startIndex, endIndex) glitch intervals. These indices
        must be in the same coordinate system as processedAll.

    extractLen : float
        Length of the nominal analysis interval in seconds.

    overlapBuffer : float
        Amount of processed data preceding hourStart.

    nSegments : int
        Number of MB subsegments. Default is 13.

    overlap : float
        Fractional overlap between consecutive MB subsegments.
        Default is 0.75.

    detrend : str or callable
        Detrending passed to scipy.signal.periodogram.

    window : str or tuple
        Window passed to scipy.signal.periodogram.

    Returns
    -------
    psdAllMB : list
        Same structure as output from computeHourlyPSDs().
        Each available channel contains one hourly PSD realization,
        so "psd" has shape (1, nFrequency).

    psdAllMBDeGlitch : list
        Same structure as psdAllMB, but using only clean MB
        subsegments. Entry is None if no clean subsegment survives.

    nSegmentsCleanAll : list
        Number of clean MB subsegments remaining for each channel.
        None for unavailable channels.
    """

    # ---------------------------------------------------------
    # Determine MB segment length and hop
    #
    # extractLen = L + (nSegments - 1) * hop
    # hop        = L * (1 - overlap)
    # ---------------------------------------------------------

    segmentLen = extractLen/(1.0 + (nSegments - 1)*(1.0 - overlap))

    segmentHop = segmentLen * (1.0 - overlap)

    psdAllMB = []
    psdAllMBDeGlitch = []
    nSegmentsCleanAll = []

    for streamNo, st in enumerate(processedAll):

        # -----------------------------------------------------
        # Missing channel
        # -----------------------------------------------------

        if st is None or len(st) == 0:
            psdAllMB.append(None)
            psdAllMBDeGlitch.append(None)
            nSegmentsCleanAll.append(None)
            continue

        if len(st) != 1:
            raise ValueError(
                f"processedAll[{streamNo}] contains {len(st)} traces. "
                "Expected one merged trace."
            )

        tr = st[0]

        samplingRate = float(tr.stats.sampling_rate)

        nSamplesBuffer = int(round(overlapBuffer * samplingRate))
        nSamplesHour = int(round(extractLen * samplingRate))

        nSamplesSegment = int(round(segmentLen * samplingRate))
        nSamplesHop = int(round(segmentHop * samplingRate))

        nSamplesExpected = nSamplesBuffer + nSamplesHour

        data = np.asarray(tr.data, dtype=np.float64)

        # ObsPy may retain one endpoint sample.
        if data.size < nSamplesExpected:
            print(f"{tr.id}: expected at least {nSamplesExpected} samples, "
                f"but found only {data.size}. Skipping this channel.")

            psdAllMB.append(None)
            psdAllMBDeGlitch.append(None)
            nSegmentsCleanAll.append(None)
            continue

        data = data[:nSamplesExpected]

        # -----------------------------------------------------
        # Compute the 13 MB subsegment PSDs once
        # -----------------------------------------------------

        psdSegments = []
        segmentHasGlitch = []
        segmentStartTimes = []

        frequencies = None

        for segmentNo in range(nSegments):

            # Start at the nominal hour, i.e. after the
            # leading overlapBuffer.
            startIndex = (nSamplesBuffer+segmentNo * nSamplesHop)
            endIndex = startIndex + nSamplesSegment
            segment = data[startIndex:endIndex]
            if segment.size != nSamplesSegment:
                raise RuntimeError(
                    f"{tr.id}, MB segment {segmentNo}: expected "
                    f"{nSamplesSegment} samples but obtained "
                    f"{segment.size}."
                )

            frequenciesNow, psdNow = periodogram(segment,fs=samplingRate,window=window,detrend=detrend,
                                                 return_onesided=True,scaling="density",)

            if frequencies is None:
                frequencies = frequenciesNow

            psdSegments.append(psdNow)

            segmentStartTimes.append(tr.stats.starttime+startIndex / samplingRate)
            # -------------------------------------------------
            # Check whether this MB segment overlaps a glitch
            # -------------------------------------------------

            if (glitchInt is None or streamNo >= len(glitchInt)or glitchInt[streamNo] is None):
                hasGlitch = False

            else:
                hasGlitch = any(startIndex <= high and endIndex > low
                    for low, high in glitchInt[streamNo]
                )

            segmentHasGlitch.append(hasGlitch)

        psdSegments = np.asarray(psdSegments,dtype=np.float64,)

        segmentHasGlitch = np.asarray(segmentHasGlitch,dtype=bool)

        # -----------------------------------------------------
        # Standard MB result: mean of all 13 segments
        # -----------------------------------------------------

        psdMB = np.mean(psdSegments, axis=0)

        psdAllMB.append({"seed_id": tr.id,"sampling_rate": samplingRate,"frequencies": frequencies,
                         "psd": psdMB[None, :],"window_start_times": [hourStart]})

        # -----------------------------------------------------
        # Deglitched MB result: mean of clean segments only
        # -----------------------------------------------------

        cleanMask = ~segmentHasGlitch
        nClean = int(np.sum(cleanMask))

        nSegmentsCleanAll.append(nClean)

        if nClean == 0:
            psdAllMBDeGlitch.append(None)

        else:
            psdMBClean = np.mean(psdSegments[cleanMask],axis=0,)

            psdAllMBDeGlitch.append({"seed_id": tr.id,"sampling_rate": samplingRate,"frequencies": frequencies,
                                     "psd": psdMBClean[None, :],"window_start_times": [hourStart],})

    return (psdAllMB,psdAllMBDeGlitch,nSegmentsCleanAll)

def initializePSHistograms(nChannels=9,psdMin=1e-30,psdMax=1e-8,nPsdBins=220,freqMin=0.0,freqMax=20.0,df=1.0 / 120.0,):
    """
    Initialize PSD histogram matrices.

    Rows:
        PSD bins

    Columns:
        frequency bins
    """

    # PSD-bin edges and centres
    psdEdges = np.logspace(np.log10(psdMin),np.log10(psdMax),nPsdBins + 1)

    psdCenters = np.sqrt(psdEdges[:-1] * psdEdges[1:])

    # Frequency-bin centres
    freqHist = np.arange(freqMin,freqMax + 0.5 * df,df,dtype=np.float64)

    # Frequency-bin edges for pcolormesh
    freqEdges = np.empty(freqHist.size + 1)

    freqEdges[1:-1] = 0.5 * (freqHist[:-1] + freqHist[1:])

    freqEdges[0] = freqHist[0] - 0.5 * df
    freqEdges[-1] = freqHist[-1] + 0.5 * df

    # Avoid a negative first edge when starting at 0 Hz
    if freqHist[0] == 0.0:
        freqEdges[0] = 0.0

    histAll = [np.zeros((nPsdBins, freqHist.size),dtype=np.uint64,)
        for _ in range(nChannels)
    ]

    return (histAll,psdEdges,psdCenters,freqHist,freqEdges)

def updatePSHistograms(histAll,psdAll,psdEdges,freqHist):
    """
    Accumulate one hour of PSD estimates into channel histograms.

    Parameters
    ----------
    histAll : list of ndarray
        Histogram matrices with shape:
            (nPsdBins, nFrequencies)

    psdAll : list
        Output from computeHourlyPSDs(). Each available entry must
        contain 'frequencies' and 'psd'.

    psdEdges : ndarray
        PSD-bin edges in linear PSD units.

    freqHist : ndarray
        Common frequency grid used by the histogram.

    Returns
    -------
    histAll : list of ndarray
        Updated histogram matrices.
    """

    nPsdBins = len(psdEdges) - 1

    for channelNo, psdResult in enumerate(psdAll):

        if psdResult is None:
            continue

        frequencies = psdResult["frequencies"]
        psdMatrix = psdResult["psd"]

        # Select the frequency range represented in freqHist.
        freqMask = ((frequencies >= freqHist[0] - 1e-10)&(frequencies <= freqHist[-1] + 1e-10))

        frequenciesUse = frequencies[freqMask]
        psdUse = psdMatrix[:, freqMask]

        if frequenciesUse.size != freqHist.size:
            raise ValueError(
                f"Channel {channelNo}: expected "
                f"{freqHist.size} frequency bins, but found "
                f"{frequenciesUse.size}."
            )

        if not np.allclose(frequenciesUse,freqHist,rtol=1e-10,atol=1e-10,):
            raise ValueError(
                f"Channel {channelNo}: frequency grid does not "
                "match the histogram frequency grid."
            )

        # psdUse shape:
        #     (60 windows, nFrequencies)
        #
        # Convert every PSD value to its row-bin index.
        binIndices = np.searchsorted(psdEdges,psdUse,side="right") - 1

        # Valid values lie inside:
        # psdEdges[0] <= PSD < psdEdges[-1]
        valid = (np.isfinite(psdUse)& (psdUse > 0.0)& (binIndices >= 0)& (binIndices < nPsdBins))

        # Accumulate all windows and frequencies without looping
        # over every PSD value in Python.
        windowIndices, freqIndices = np.nonzero(valid)

        rowIndices = binIndices[windowIndices,freqIndices]

        np.add.at(histAll[channelNo],(rowIndices, freqIndices),1)

    return histAll

def plotPSDHistogram(histAll,channelNo,freqHist,psdEdges,ax,seedID=None,normalize=True,logCounts=False):
    """
    Plot the accumulated PSD histogram for one channel.

    Parameters
    ----------
    histAll : list of ndarray
        One histogram matrix per channel, with shape
        (nPsdBins, nFrequencies).
    channelNo : int
        Channel index to plot.
    freqHist : ndarray
        Frequency corresponding to each histogram column.
    psdEdges : ndarray
        PSD-bin edges.
    seedID : str or None
        Channel label for the title.
    normalize : bool
        If True, normalize every frequency column independently
        to percentage occurrence.
    logCounts : bool
        If True and normalize=False, plot raw counts with logarithmic
        colour normalization.
    """

    hist = histAll[channelNo].astype(np.float64)

    if hist.size == 0 or not np.any(hist):
        print(f"No histogram counts available for channel {channelNo}.")
        return

    if normalize:
        countsPerFrequency = hist.sum(axis=0, keepdims=True)

        histPlot = np.divide(hist,countsPerFrequency,out=np.zeros_like(hist),where=countsPerFrequency > 0)

        histPlot *= 100.0
        colorLabel = "Occurrence (%)"
        norm = None

    else:
        histPlot = hist
        colorLabel = "Count"

        if logCounts:
            positiveValues = histPlot[histPlot > 0]

            norm = LogNorm(vmin=max(1.0, positiveValues.min()),vmax=positiveValues.max(),)
        else:
            norm = None

    #fig, ax = plt.subplots(figsize=(11, 6))

    mesh = ax.pcolormesh(freqHist,psdEdges,histPlot,shading="auto",norm=norm)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(freqHist[0], freqHist[-1])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Acceleration PSD ($\mathrm{m^2/s^4/Hz}$)")
    ax.set_ylim(10**-18,10**-10)

    title = "PSD histogram"
    if seedID is not None:
        title += f": {seedID}"

    ax.set_title(title)

    
    # Get the figure belonging to the supplied axis
    fig = ax.figure

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(colorLabel)

    return mesh

def percentilesFromHistogram(hist,psdEdges,percentiles=(10.0, 50.0, 90.0)):
    """
    Estimate PSD percentiles independently at every frequency.

    Parameters
    ----------
    hist : ndarray
        Histogram counts with shape:
            (nPsdBins, nFrequencies)

    psdEdges : ndarray
        PSD-bin edges with length nPsdBins + 1.

    percentiles : sequence of float
        Requested percentiles between 0 and 100.

    Returns
    -------
    percentileValues : ndarray
        Shape:
            (nPercentiles, nFrequencies)

        Columns with no histogram counts are returned as NaN.
    """

    hist = np.asarray(hist)
    percentiles = np.asarray(percentiles, dtype=float)

    if hist.ndim != 2:
        raise ValueError("hist must be a two-dimensional array.")

    if len(psdEdges) != hist.shape[0] + 1:
        raise ValueError("len(psdEdges) must equal hist.shape[0] + 1.")

    if np.any((percentiles < 0.0) | (percentiles > 100.0)):
        raise ValueError("Percentiles must lie between 0 and 100.")

    cumulativeCounts = np.cumsum(hist, axis=0)
    totalCounts = cumulativeCounts[-1, :]

    psdCenters = np.sqrt(psdEdges[:-1] * psdEdges[1:])

    percentileValues = np.full((len(percentiles), hist.shape[1]),np.nan,dtype=float)

    validFrequency = totalCounts > 0

    for percentileNo, percentile in enumerate(percentiles):

        thresholds = (percentile / 100.0) * totalCounts

        # First PSD bin where the cumulative count reaches
        # the requested percentile threshold.
        binIndices = np.argmax(cumulativeCounts >= thresholds[None, :],axis=0)

        percentileValues[percentileNo,validFrequency,] = psdCenters[binIndices[validFrequency]]

    return percentileValues

def getHistogramPercentiles(hist,psdEdges,percentiles=(10.0, 50.0, 90.0)):
    """
    Compute percentile curves from a histogram.

    Parameters
    ----------
    hist : ndarray
        Histogram matrix with shape (nPsdBins, nFreqBins)
    psdEdges : ndarray
        PSD bin edges, length nPsdBins + 1
    percentiles : tuple
        Requested percentiles

    Returns
    -------
    percentileValues : ndarray
        Shape (nPercentiles, nFreqBins)
    """

    hist = np.asarray(hist)

    cumulativeCounts = np.cumsum(hist, axis=0)
    totalCounts = cumulativeCounts[-1, :]

    psdCenters = np.sqrt(psdEdges[:-1] * psdEdges[1:])

    percentileValues = np.full((len(percentiles), hist.shape[1]),np.nan,dtype=float)

    validFrequency = totalCounts > 0

    for iPer, percentile in enumerate(percentiles):
        threshold = (percentile / 100.0) * totalCounts

        binIndices = np.argmax(cumulativeCounts >= threshold[None, :],axis=0)

        percentileValues[iPer, validFrequency] = psdCenters[binIndices[validFrequency]]

    return percentileValues

def smoothPSDDB(freq, psd, smoothHz=0.05):
    """
    Smooth a positive PSD curve along frequency in dB space.

    smoothHz is the Gaussian sigma in Hz.
    """

    freq = np.asarray(freq)
    psd = np.asarray(psd)

    out = psd.copy().astype(float)

    valid = (np.isfinite(freq) & np.isfinite(psd) & (psd > 0))

    if np.sum(valid) < 3:
        return out

    fNow = freq[valid]
    psdDB = 10.0 * np.log10(psd[valid])

    df = np.median(np.diff(fNow))

    sigmaBins = (smoothHz / df)

    if sigmaBins > 0:

        psdDB = gaussian_filter1d(psdDB,sigma=sigmaBins,mode="nearest")

    out[valid] = 10.0**(psdDB/10.0)

    return out

def smoothPSDInFrequency(freq,psd,smoothHz=0.05):

    freq = np.asarray(freq)
    psd = np.asarray(psd)

    out = np.full_like(psd,np.nan,dtype=float)
    valid = (np.isfinite(freq) & np.isfinite(psd) & (psd > 0))

    if np.sum(valid) < 3:
        return psd.copy()

    f = freq[valid]
    y = 10.0 * np.log10(psd[valid])

    # Assumes approximately uniform frequency spacing
    df = np.median(np.diff(f))

    sigmaBins = (smoothHz/df)

    ySmooth = gaussian_filter1d(y,sigma=sigmaBins,mode="nearest")

    out[valid] = (10.0**(ySmooth / 10.0))

    return out

def addPercentileBand(ax,freqHist,hist,psdEdges,color,label=None,alpha=0.20,lineWidth=2.0,lineStyle="-",
                      percentiles=(10.0, 50.0, 90.0),zorder_fill=1,zorder_line=2,smooth=False,smoothHz=0.05):
    """
    Add a 10-90 percentile shaded band and a 50th percentile line
    from one histogram to an existing axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis on which to draw
    freqHist : ndarray
        Frequency array
    hist : ndarray
        Histogram matrix, shape (nPsdBins, nFreqBins)
    psdEdges : ndarray
        PSD bin edges
    color : str
        Matplotlib color
    label : str or None
        Legend label for the 50th percentile line
    alpha : float
        Transparency of the shaded band
    """

    percentileValues = getHistogramPercentiles(hist=hist,psdEdges=psdEdges,percentiles=percentiles)

    p10 = percentileValues[0]
    p50 = percentileValues[1]
    p90 = percentileValues[2]
    
    # ---------------------------------------------------------
    # Smooth percentile curves in dB along frequency
    # ---------------------------------------------------------

    if smooth:
        p10 = smoothPSDDB(freqHist,p10,smoothHz=smoothHz)
        p50 = smoothPSDDB(freqHist,p50,smoothHz=smoothHz)
        p90 = smoothPSDDB(freqHist,p90,smoothHz=smoothHz)


    # mask NaNs so fill_between and plot behave nicely
    valid = np.isfinite(p10) & np.isfinite(p50) & np.isfinite(p90)

    if not np.any(valid):
        return

    ax.fill_between(freqHist[valid],p10[valid],p90[valid],color=color,alpha=alpha,linewidth=0.0,zorder=zorder_fill)

    ax.plot(freqHist[valid],p50[valid],color=color,linewidth=lineWidth,linestyle=lineStyle,label=label,zorder=zorder_line)


def plotPercentileSubplots(histAll,freqHist,psdEdges,subplotSpecs,ncols=2,figsize=(12, 8),xlim=(0.1, 20.0),ylim=None,
                           xlabel="Frequency (Hz)",ylabel=r"Acceleration PSD ($\mathrm{m^2/s^4/Hz}$)",sharex=True,
                           sharey=True,legend=True, smooth=False,smoothHz=0.05):
    """
    Plot percentile bands in a flexible subplot layout.

    Parameters
    ----------
    histAll : list
        List of histogram matrices
    freqHist : ndarray
        Frequency array
    psdEdges : ndarray
        PSD bin edges
    subplotSpecs : list of dict
        Each element defines one subplot, e.g.

        {
            "title": "Example subplot",
            "items": [
                {"hist_index": 0, "color": "tab:blue", "label": "Surface"},
                {"hist_index": 9, "color": "tab:orange", "label": "Underground"},
            ]
        }

    ncols : int
        Number of columns in subplot layout
    """

    nSubplots = len(subplotSpecs)
    nrows = int(np.ceil(nSubplots / ncols))

    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize,sharex=sharex,sharey=sharey)

    # make axes always iterable
    axes = np.atleast_1d(axes).ravel()

    for iSub, spec in enumerate(subplotSpecs):
        ax = axes[iSub]

        title = spec.get("title", "")
        items = spec.get("items", [])

        for item in items:
            hist_index = item["hist_index"]
            color = item.get("color", None)
            label = item.get("label", None)
            alpha = item.get("alpha", 0.20)
            lineWidth = item.get("lineWidth", 2.0)
            lineStyle = item.get("lineStyle", "-")

            addPercentileBand(ax=ax,freqHist=freqHist,hist=histAll[hist_index],psdEdges=psdEdges,color=color,
                              label=label,alpha=alpha,lineWidth=lineWidth,lineStyle=lineStyle,
                              percentiles=(10.0, 50.0,90.0),zorder_fill=1,zorder_line=2,smooth=smooth,smoothHz=smoothHz)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(title)
        ax.grid(True, which="both")

        if legend:
            ax.legend()

    # hide unused axes
    for j in range(nSubplots, len(axes)):
        axes[j].set_visible(False)

    # common labels
    for ax in axes[:nSubplots]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    for a in axes:
        formatter = ticker.LogFormatter(base=10.0, labelOnlyBase=False)
        a.xaxis.set_major_formatter(formatter)
        a.xaxis.set_minor_formatter(formatter)

    fig.tight_layout()
    #plt.show()
    
    return fig,axes
    
def getRawBoreholeHorizontals(streamAll, hourStart,extractLen=3600,bufferSec=180.0,indH1=9,indH2=10):
    """
    Extract and align the raw H1 and H2 borehole channels.

    Parameters
    ----------
    streamAll : list
        Output from loadStreams().

    hourStart : obspy.UTCDateTime
        Beginning of the actual analysis hour.

    extractLen : float
        Length of the analysis interval in seconds.

    bufferSec : float
        Extra raw data retained before and after the analysis hour.

    indH1, indH2 : int
        Indices of the borehole horizontal channels in streamAll.

    Returns
    -------
    rawData : dict or None
        Dictionary containing aligned raw H1/H2 counts and timing
        information. Returns None if the required data are unavailable.
    """

    # ---------------------------------------------------------
    # 1. Get the two raw streams
    # ---------------------------------------------------------
    stH1 = streamAll[indH1]
    stH2 = streamAll[indH2]

    if stH1 is None or len(stH1) == 0:
        print("No BH1 data.")
        return None

    if stH2 is None or len(stH2) == 0:
        print("No BH2 data.")
        return None

    # Work on copies so streamAll is not modified.
    stH1 = stH1.copy()
    stH2 = stH2.copy()

    # ---------------------------------------------------------
    # 2. Requested interval
    # ---------------------------------------------------------
    readStart = hourStart - bufferSec
    readEnd   = hourStart + extractLen + bufferSec

    stH1.trim(starttime=readStart,endtime=readEnd,nearest_sample=False,pad=False)

    stH2.trim(starttime=readStart,endtime=readEnd,nearest_sample=False,pad=False)

    if len(stH1) == 0 or len(stH2) == 0:
        print("BH1 or BH2 does not overlap requested time interval.")
        return None

    # ---------------------------------------------------------
    # 3. For the moment require one continuous Trace per channel
    # ---------------------------------------------------------
    if len(stH1) != 1 or len(stH2) != 1:
        print("Gap or multiple traces present in BH1/BH2.")
        return None

    trH1 = stH1[0]
    trH2 = stH2[0]

    # ---------------------------------------------------------
    # 4. Check sampling rates
    # ---------------------------------------------------------
    fsH1 = float(trH1.stats.sampling_rate)
    fsH2 = float(trH2.stats.sampling_rate)

    if not np.isclose(fsH1, fsH2):
        raise ValueError(f"Sampling rates differ: BH1={fsH1} Hz, BH2={fsH2} Hz")

    fs = fsH1

    # ---------------------------------------------------------
    # 5. Determine the common time interval
    # ---------------------------------------------------------
    commonStart = max(trH1.stats.starttime,trH2.stats.starttime,readStart)

    commonEnd = min(trH1.stats.endtime,trH2.stats.endtime,readEnd)

    if commonEnd <= commonStart:
        print("No common BH1/BH2 time interval.")
        return None

    trH1.trim(starttime=commonStart,endtime=commonEnd,nearest_sample=True)

    trH2.trim(starttime=commonStart,endtime=commonEnd,nearest_sample=True)

    # ---------------------------------------------------------
    # 6. Convert to ordinary NumPy arrays
    # ---------------------------------------------------------
    nSamples = min(trH1.stats.npts, trH2.stats.npts)

    H1 = np.asarray(trH1.data[:nSamples], dtype=np.float64)
    H2 = np.asarray(trH2.data[:nSamples], dtype=np.float64)

    time = np.arange(nSamples) / fs

    return {"H1": H1,"H2": H2,"time": time,"fs": fs,"startTime": commonStart,"hourStart": UTCDateTime(hourStart),
            "hourEnd": UTCDateTime(hourStart) + extractLen}

def getRawBoreholeHorizontalsBSDL(streamAll, hourStart,extractLen=3600,bufferSec=180.0,indH1=9):
    """
    Extract and align the raw H1 and H2 borehole channels.

    Parameters
    ----------
    streamAll : list
        Output from loadStreams().

    hourStart : obspy.UTCDateTime
        Beginning of the actual analysis hour.

    extractLen : float
        Length of the analysis interval in seconds.

    bufferSec : float
        Extra raw data retained before and after the analysis hour.

    indH1, indH2 : int
        Indices of the borehole horizontal channels in streamAll.

    Returns
    -------
    rawData : dict or None
        Dictionary containing aligned raw H1/H2 counts and timing
        information. Returns None if the required data are unavailable.
    """

    # ---------------------------------------------------------
    # 1. Get the two raw streams
    # ---------------------------------------------------------
    stH1 = streamAll[indH1]

    if stH1 is None or len(stH1) == 0:
        print("No BH1 data.")
        return None

    # Work on copies so streamAll is not modified.
    stH1 = stH1.copy()

    # ---------------------------------------------------------
    # 2. Requested interval
    # ---------------------------------------------------------
    readStart = hourStart - bufferSec
    readEnd   = hourStart + extractLen + bufferSec

    stH1.trim(starttime=readStart,endtime=readEnd,nearest_sample=False,pad=False)

    if len(stH1) == 0:
        print("BH2 does not overlap requested time interval.")
        return None

    # ---------------------------------------------------------
    # 3. For the moment require one continuous Trace per channel
    # ---------------------------------------------------------
    if len(stH1) != 1:
        print("Gap or multiple traces present in BH2.")
        return None

    trH1 = stH1[0]

    # ---------------------------------------------------------
    # 4. Check sampling rates
    # ---------------------------------------------------------
    fsH1 = float(trH1.stats.sampling_rate)

    fs = fsH1

    # ---------------------------------------------------------
    # 5. Determine the common time interval
    # ---------------------------------------------------------
    commonStart = max(trH1.stats.starttime,trH1.stats.starttime,readStart)

    commonEnd = min(trH1.stats.endtime,trH1.stats.endtime,readEnd)

    if commonEnd <= commonStart:
        print("No common BH1/BH2 time interval.")
        return None

    trH1.trim(starttime=commonStart,endtime=commonEnd,nearest_sample=True)
    # ---------------------------------------------------------
    # 6. Convert to ordinary NumPy arrays
    # ---------------------------------------------------------
    nSamples = min(trH1.stats.npts, trH1.stats.npts)

    H1 = np.asarray(trH1.data[:nSamples], dtype=np.float64)
 
    time = np.arange(nSamples) / fs

    return {"H1": H1,"time": time,"fs": fs,"startTime": commonStart,"hourStart": UTCDateTime(hourStart),
            "hourEnd": UTCDateTime(hourStart) + extractLen}

def computeHorizontalDeparture(rawBH, baselineMethod="median"):
    """
    Compute baseline-subtracted raw horizontal counts and their
    combined excursion magnitude.

    Parameters
    ----------
    rawBH : dict
        Output from getRawBoreholeHorizontals().

    baselineMethod : str
        Method used to estimate the normal raw-count baseline.
        Currently only "median" is implemented.

    Returns
    -------
    depBH : dict
        Dictionary containing baseline values, baseline-subtracted
        H1/H2 traces, and combined horizontal departure.
    """

    if rawBH is None:
        return None

    H1 = np.asarray(rawBH["H1"], dtype=np.float64)
    H2 = np.asarray(rawBH["H2"], dtype=np.float64)

    if baselineMethod == "median":

        baseH1 = np.nanmedian(H1)
        baseH2 = np.nanmedian(H2)

    else:
        raise ValueError(f"Unknown baselineMethod: {baselineMethod}")

    # Baseline-subtracted raw counts
    dH1 = H1 - baseH1
    dH2 = H2 - baseH2

    # Combined horizontal departure
    R = np.sqrt(dH1**2 + dH2**2)

    return {"dH1": dH1,"dH2": dH2,"R": R,"baselineH1": baseH1,"baselineH2": baseH2,"time": rawBH["time"],
            "fs": rawBH["fs"],"startTime": rawBH["startTime"],"hourStart": rawBH["hourStart"],
            "hourEnd": rawBH["hourEnd"]}

def computeHorizontalDepartureBSDL(rawBH, baselineMethod="median"):
    """
    Compute baseline-subtracted raw horizontal counts and their
    combined excursion magnitude.

    Parameters
    ----------
    rawBH : dict
        Output from getRawBoreholeHorizontals().

    baselineMethod : str
        Method used to estimate the normal raw-count baseline.
        Currently only "median" is implemented.

    Returns
    -------
    depBH : dict
        Dictionary containing baseline values, baseline-subtracted
        H1/H2 traces, and combined horizontal departure.
    """

    if rawBH is None:
        return None

    H1 = np.asarray(rawBH["H1"], dtype=np.float64)

    if baselineMethod == "median":

        baseH1 = np.nanmedian(H1)

    else:
        raise ValueError(f"Unknown baselineMethod: {baselineMethod}")

    # Baseline-subtracted raw counts
    dH1 = H1 - baseH1

    # Combined horizontal departure
    R = np.sqrt(dH1**2)

    return {"dH1": dH1,"R": R,"baselineH1": baseH1,"time": rawBH["time"],
            "fs": rawBH["fs"],"startTime": rawBH["startTime"],"hourStart": rawBH["hourStart"],
            "hourEnd": rawBH["hourEnd"]}

def detectExcursions(depBH,thresholdHigh=1.0e6,thresholdLow=3.0e5,quietSec=10.0,preSec=5.0):
    """
    Detect raw-count excursions using hysteresis.

    An event starts when R exceeds thresholdHigh.

    It ends when R remains below thresholdLow continuously
    for quietSec seconds.

    Parameters
    ----------
    depBH : dict
        Output from computeHorizontalDeparture().

    thresholdHigh : float
        Trigger threshold in counts.

    thresholdLow : float
        Recovery threshold in counts.

    quietSec : float
        Required continuous quiet time before declaring the
        event finished.

    Returns
    -------
    events : list of tuple
        List of (startIndex, endIndex) pairs.
    """

    R = depBH["R"]
    fs = depBH["fs"]

    nQuiet = int(round(quietSec * fs))
    nPre = int(round(preSec * fs))

    events = []

    inEvent = False
    startIndex = None
    quietCount = 0

    for i in range(len(R)):

        if not inEvent:

            if R[i] >= thresholdHigh:
                inEvent = True
                startIndex = i
                quietCount = 0

        else:

            if R[i] < thresholdLow:
                quietCount += 1

                if quietCount >= nQuiet:

                    endIndex = i - nQuiet + 1

                    startBuffered = max(0, startIndex - nPre)

                    events.append((startBuffered, endIndex))

                    inEvent = False
                    startIndex = None
                    quietCount = 0

            else:
                quietCount = 0

    # Event continuing until the end of the chunk
    if inEvent:
        startBuffered = max(0, startIndex - nPre)
        events.append((startBuffered, len(R) - 1))
        
    # Because the pre-event buffer can cause neighbouring events to overlap,
    # merge overlapping intervals.
    if len(events) > 1:

        mergedEvents = [events[0]]

        for start, end in events[1:]:

            prevStart, prevEnd = mergedEvents[-1]

            if start <= prevEnd:
                mergedEvents[-1] = (prevStart, max(prevEnd, end))
            else:
                mergedEvents.append((start, end))

        events = mergedEvents

    return events

def autoscaleAllLinesToVisibleX(ax, marginFraction=0.05):
    xmin, xmax = ax.get_xlim()

    visibleValues = []

    for line in ax.get_lines():
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())

        mask = (
            (x >= xmin)
            & (x <= xmax)
            & np.isfinite(y)
        )

        if np.any(mask):
            visibleValues.append(y[mask])

    if not visibleValues:
        return

    visibleValues = np.concatenate(visibleValues)

    ymin = visibleValues.min()
    ymax = visibleValues.max()

    if ymax > ymin:
        margin = marginFraction * (ymax - ymin)
    else:
        margin = (
            marginFraction * abs(ymax)
            if ymax != 0
            else 1.0
        )

    ax.set_ylim(ymin - margin, ymax + margin)
    
def refineGlitchesDeconvolved(processedStream,eventsPass,indH1=9,indH2=10,thresholdHigh=None,thresholdLow=None,
                              thresholdHighFactor=12.0,thresholdLowFactor=4.0,quietSec=1.0,padSec=0.5):
    """
    An older version, works but not with trains
    Refine broad raw-count glitch intervals using the deconvolved
    horizontal acceleration data.

    The search is performed ONLY inside intervals already identified
    by the raw-count detector.

    Parameters
    ----------
    processedStream : list of obspy.Stream
        Response-corrected streams returned by prepareStream().

    eventsPass : list of tuple
        Broad glitch intervals from the raw-count detector, expressed
        as (startIndex, endIndex) relative to the analysis hour.

    indH1, indH2 : int
        Indices of the two borehole horizontal components in
        processedStream.

    thresholdHigh, thresholdLow : float or None
        Absolute thresholds in acceleration units. If None, robust
        thresholds are estimated from the deconvolved horizontal
        amplitude.

    thresholdHighFactor : float
        High threshold in units of robust background scale.

    thresholdLowFactor : float
        Low/recovery threshold in units of robust background scale.

    quietSec : float
        Number of seconds continuously below thresholdLow required
        to end a refined glitch.

    padSec : float
        Padding added before and after each refined glitch.

    Returns
    -------
    refinedEvents : list of tuple
        Refined (startIndex, endIndex) glitch intervals, in the same
        sample-coordinate system as eventsPass.
    """

    # ---------------------------------------------------------
    # 1. Get processed horizontal traces
    # ---------------------------------------------------------
    stH1 = processedStream[indH1]
    stH2 = processedStream[indH2]

    if stH1 is None or len(stH1) == 0:
        return []

    if stH2 is None or len(stH2) == 0:
        return []

    if len(stH1) != 1 or len(stH2) != 1:
        raise ValueError("Expected one processed trace for each horizontal channel.")

    trH1 = stH1[0]
    trH2 = stH2[0]

    fs1 = float(trH1.stats.sampling_rate)
    fs2 = float(trH2.stats.sampling_rate)

    if not np.isclose(fs1, fs2):
        raise ValueError(
            f"Sampling rates differ: H1={fs1} Hz, H2={fs2} Hz"
        )

    fs = fs1

    H1 = np.asarray(trH1.data, dtype=np.float64)
    H2 = np.asarray(trH2.data, dtype=np.float64)

    nSamples = min(len(H1), len(H2))
    H1 = H1[:nSamples]
    H2 = H2[:nSamples]

    # ---------------------------------------------------------
    # 2. Combined horizontal acceleration amplitude
    # ---------------------------------------------------------
    R = np.sqrt(H1**2 + H2**2)

    # ---------------------------------------------------------
    # 3. Robust global background scale
    #
    # Median is usually approximately zero after response removal,
    # but subtract it anyway.
    # ---------------------------------------------------------
    medianR = np.nanmedian(R)
    madR = np.nanmedian(np.abs(R - medianR))

    # Convert MAD to approximately sigma-equivalent scale.
    sigmaRobust = 1.4826 * madR

    if sigmaRobust <= 0 or not np.isfinite(sigmaRobust):
        #raise ValueError("Could not estimate a valid robust background scale.")
        return []

    if thresholdHigh is None:
        thresholdHigh = medianR + thresholdHighFactor * sigmaRobust

    if thresholdLow is None:
        thresholdLow = medianR + thresholdLowFactor * sigmaRobust

    nQuiet = max(1, int(round(quietSec * fs)))
    nPad = int(round(padSec * fs))

    refinedEvents = []

    # ---------------------------------------------------------
    # 4. Search only inside broad raw-count events
    # ---------------------------------------------------------
    for broadStart, broadEnd in eventsPass:

        broadStartProc = int(broadStart)
        broadEndProc = int(broadEnd)

        # Clip safely
        broadStartProc = max(0, broadStartProc)
        broadEndProc = min(nSamples - 1, broadEndProc)

        if broadEndProc <= broadStartProc:
            continue

        inEvent = False
        startIndex = None
        quietCount = 0

        for i in range(broadStartProc, broadEndProc + 1):

            if not inEvent:

                if R[i] >= thresholdHigh:
                    inEvent = True
                    startIndex = i
                    quietCount = 0

            else:

                if R[i] < thresholdLow:
                    quietCount += 1

                    if quietCount >= nQuiet:

                        endIndex = i - nQuiet + 1

                        # Apply small guard interval
                        startPadded = max(broadStartProc,startIndex - nPad)

                        endPadded = min(broadEndProc,endIndex + nPad)

                        refinedEvents.append((startPadded, endPadded))

                        inEvent = False
                        startIndex = None
                        quietCount = 0

                else:
                    quietCount = 0

        # If spike remains active to the end of the broad raw event
        if inEvent:

            startPadded = max(broadStartProc,startIndex - nPad)

            endPadded = broadEndProc

            refinedEvents.append((startPadded, endPadded))
        
        # Padding can cause neighbouring G2 intervals to overlap.
        # In that case, treat them as one continuous contaminated
        # interval.
        # ---------------------------------------------------------
        if len(refinedEvents) > 1:

            mergedEvents = [refinedEvents[0]]

            for start, end in refinedEvents[1:]:

                prevStart, prevEnd = mergedEvents[-1]

                if start <= prevEnd:
                    mergedEvents[-1] = (prevStart,max(prevEnd, end))
                else:
                    mergedEvents.append((start, end))

            refinedEvents = mergedEvents

    return refinedEvents

def _read_trace(processedStream, index, label):
    if index is None:
        return None
    if not (0 <= int(index) < len(processedStream)):
        raise IndexError(f"{label} index {index} is outside processedStream")
    st = processedStream[int(index)]
    if st is None or len(st) == 0:
        return None
    if len(st) != 1:
        raise ValueError(f"Expected one processed trace for {label} (index {index})")
    tr = st[0]
    fs = float(tr.stats.sampling_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Invalid sampling rate for {label}")
    return tr

def _find_impulses(R, fs, eventsPass, *, thresholdHigh, thresholdLow,
                   thresholdHighFactor, thresholdLowFactor, quietSec,
                   padSec, shortSec, backgroundSec, sigmaFloorFactor):
    #Apply the original robust impulsiveness detector to one |component|.
    R = np.asarray(R, dtype=np.float64)
    nSamples = R.size
    if nSamples < 2 or not np.all(np.isfinite(R)):
        return []

    nShort = max(1, int(round(shortSec * fs)))
    Rshort = np.sqrt(uniform_filter1d(R**2, size=nShort, mode="nearest"))

    nBackground = max(3, int(round(backgroundSec * fs)))
    if nBackground % 2 == 0:
        nBackground += 1

    localMedian = median_filter(Rshort, size=nBackground, mode="nearest")
    localMAD = median_filter(np.abs(Rshort-localMedian),
                             size=nBackground, mode="nearest")
    localSigma = 1.4826 * localMAD

    globalMedian = np.nanmedian(Rshort)
    globalMAD = np.nanmedian(np.abs(Rshort-globalMedian))
    globalSigma = 1.4826 * globalMAD
    if globalSigma <= 0 or not np.isfinite(globalSigma):
        return []

    sigmaUse = np.maximum(localSigma, sigmaFloorFactor*globalSigma)
    Z = (Rshort-localMedian) / sigmaUse

    zHigh = thresholdHighFactor if thresholdHigh is None else float(thresholdHigh)
    zLow = thresholdLowFactor if thresholdLow is None else float(thresholdLow)
    nQuiet = max(1, int(round(quietSec*fs)))
    nPad = max(0, int(round(padSec*fs)))
    refinedEvents = []

    for broadStart, broadEnd in eventsPass:
        broadStartProc = max(0, int(broadStart))
        broadEndProc = min(nSamples-1, int(broadEnd))
        if broadEndProc <= broadStartProc:
            continue
        inEvent = False
        startIndex = None
        quietCount = 0
        for i in range(broadStartProc, broadEndProc+1):
            if not inEvent:
                if Z[i] >= zHigh:
                    inEvent = True
                    startIndex = i
                    quietCount = 0
            elif Z[i] < zLow:
                quietCount += 1
                if quietCount >= nQuiet:
                    endIndex = i-nQuiet+1
                    startPadded = max(broadStartProc, startIndex-nPad)
                    endPadded = min(broadEndProc, endIndex+nPad)
                    refinedEvents.append((startPadded, endPadded))
                    inEvent = False
                    startIndex = None
                    quietCount = 0
            else:
                quietCount = 0
        if inEvent:
            refinedEvents.append((max(broadStartProc, startIndex-nPad),
                                  broadEndProc))
    return refinedEvents

def _merge_events(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda p: (p[0], p[1]))
    out = [tuple(map(int, intervals[0]))]
    for a, b in intervals[1:]:
        p, q = out[-1]
        if a <= q+1:
            out[-1] = (p, max(q, int(b)))
        else:
            out.append((int(a), int(b)))
    return out

def _empty_result():
    return {
        "intervals": [],
        "horizontal_intervals": [],
        "vertical_intervals": [],
        "bh1_intervals": [],
        "bh2_intervals": [],
        "component_intervals": {},
    }

def _refine(processedStream, eventsPass, horizontalIndices, indV, *,
            thresholdHigh, thresholdLow, thresholdHighFactor,
            thresholdLowFactor, quietSec, padSec, shortSec,
            backgroundSec, sigmaFloorFactor, returnDiagnostics):
    if eventsPass is None or len(eventsPass) == 0:
        result = _empty_result()
        return result if returnDiagnostics else []

    # IMPORTANT: run each horizontal individually; do NOT combine amplitudes.
    # Missing components contribute no detections, but must not disable the
    # other components. The same applies to vertical-only detections.
    names_and_indices = [(f"bh{j+1}", ind)
                         for j, ind in enumerate(horizontalIndices)]
    names_and_indices.append(("vertical", indV))

    traces = []
    for name, ind in names_and_indices:
        tr = _read_trace(processedStream, ind, name)
        if tr is not None:
            traces.append((name, tr))

    if not traces:
        result = _empty_result()
        return result if returnDiagnostics else []

    fs = float(traces[0][1].stats.sampling_rate)
    reference_start = traces[0][1].stats.starttime
    for name, tr in traces[1:]:
        fs_now = float(tr.stats.sampling_rate)
        if not np.isclose(fs, fs_now, rtol=1e-6, atol=0.0):
            raise ValueError(f"G2 sample rates differ for {name}")
        if abs(float(tr.stats.starttime-reference_start)) > 0.51/fs:
            raise ValueError(
                "G2 trace start times differ: align the processed traces first"
            )

    params = dict(thresholdHigh=thresholdHigh, thresholdLow=thresholdLow,
                  thresholdHighFactor=thresholdHighFactor,
                  thresholdLowFactor=thresholdLowFactor, quietSec=quietSec,
                  padSec=padSec, shortSec=shortSec,
                  backgroundSec=backgroundSec,
                  sigmaFloorFactor=sigmaFloorFactor)

    intervals_by_component = {}
    for name, tr in traces:
        # The individual signal, not a combined 2D/3D vector amplitude.
        R = np.abs(np.asarray(tr.data, dtype=np.float64))
        intervals_by_component[name] = _merge_events(
            _find_impulses(R, fs, eventsPass, **params)
        )

    h1 = intervals_by_component.get("bh1", [])
    h2 = intervals_by_component.get("bh2", [])
    vz = intervals_by_component.get("vertical", [])
    horizontal = _merge_events(h1 + h2)

    result = {
        "intervals": _merge_events(h1 + h2 + vz),
        "horizontal_intervals": horizontal,
        "vertical_intervals": vz,
        "bh1_intervals": h1,
        "bh2_intervals": h2,
        "component_intervals": intervals_by_component,
    }
    return result if returnDiagnostics else result["intervals"]


def refineGlitchesDeconvolvedNew(
        processedStream, eventsPass, indH1=9, indH2=10,
        thresholdHigh=None, thresholdLow=None,
        thresholdHighFactor=12.0, thresholdLowFactor=4.0,
        quietSec=1.0, padSec=0.5, shortSec=0.25,
        backgroundSec=10.0, sigmaFloorFactor=0.1,
        indV=None, returnDiagnostics=False):
    """G2 detection independently on BH1, BH2 and (optional) BHZ.

    Independently detect events on each component using the original robust
    short-RMS/MAD statistic and hysteretic windowing. Return the merged union.
    Set indV explicitly to the appropriate vertical-channel index.
    """
    return _refine(processedStream, eventsPass, [indH1, indH2], indV,
                   thresholdHigh=thresholdHigh, thresholdLow=thresholdLow,
                   thresholdHighFactor=thresholdHighFactor,
                   thresholdLowFactor=thresholdLowFactor, quietSec=quietSec,
                   padSec=padSec, shortSec=shortSec,
                   backgroundSec=backgroundSec,
                   sigmaFloorFactor=sigmaFloorFactor,
                   returnDiagnostics=returnDiagnostics)


def refineGlitchesDeconvolvedBSDLNew(
        processedStream, eventsPass, indH1=9,
        thresholdHigh=None, thresholdLow=None,
        thresholdHighFactor=12.0, thresholdLowFactor=4.0,
        quietSec=1.0, padSec=0.5, shortSec=0.25,
        backgroundSec=10.0, sigmaFloorFactor=0.1,
        indV=None, returnDiagnostics=False):
    """BSDL: independent G2 on its working BH2 and optional vertical."""
    return _refine(processedStream, eventsPass, [None, indH1], indV,
                   thresholdHigh=thresholdHigh, thresholdLow=thresholdLow,
                   thresholdHighFactor=thresholdHighFactor,
                   thresholdLowFactor=thresholdLowFactor, quietSec=quietSec,
                   padSec=padSec, shortSec=shortSec,
                   backgroundSec=backgroundSec,
                   sigmaFloorFactor=sigmaFloorFactor,
                   returnDiagnostics=returnDiagnostics)

def refineGlitchesDeconvolvedBSDL(processedStream,eventsPass,indH1=9,thresholdHigh=None,thresholdLow=None,
                              thresholdHighFactor=12.0,thresholdLowFactor=4.0,quietSec=1.0,padSec=0.5):
    """
    Refine broad raw-count glitch intervals using the deconvolved
    horizontal acceleration data.

    The search is performed ONLY inside intervals already identified
    by the raw-count detector.

    Parameters
    ----------
    processedStream : list of obspy.Stream
        Response-corrected streams returned by prepareStream().

    eventsPass : list of tuple
        Broad glitch intervals from the raw-count detector, expressed
        as (startIndex, endIndex) relative to the analysis hour.

    indH1, indH2 : int
        Indices of the two borehole horizontal components in
        processedStream.

    thresholdHigh, thresholdLow : float or None
        Absolute thresholds in acceleration units. If None, robust
        thresholds are estimated from the deconvolved horizontal
        amplitude.

    thresholdHighFactor : float
        High threshold in units of robust background scale.

    thresholdLowFactor : float
        Low/recovery threshold in units of robust background scale.

    quietSec : float
        Number of seconds continuously below thresholdLow required
        to end a refined glitch.

    padSec : float
        Padding added before and after each refined glitch.

    Returns
    -------
    refinedEvents : list of tuple
        Refined (startIndex, endIndex) glitch intervals, in the same
        sample-coordinate system as eventsPass.
    """

    # ---------------------------------------------------------
    # 1. Get processed horizontal traces
    # ---------------------------------------------------------
    stH1 = processedStream[indH1]

    if stH1 is None or len(stH1) == 0:
        return []

    if len(stH1) != 1:
        raise ValueError("Expected one processed trace for horizontal channel.")

    trH1 = stH1[0]

    fs1 = float(trH1.stats.sampling_rate)

    fs = fs1

    H1 = np.asarray(trH1.data, dtype=np.float64)

    nSamples = min(len(H1), len(H1))
    H1 = H1[:nSamples]

    # ---------------------------------------------------------
    # 2. Combined horizontal acceleration amplitude
    # ---------------------------------------------------------
    R = np.sqrt(H1**2)

    # ---------------------------------------------------------
    # 3. Robust global background scale
    #
    # Median is usually approximately zero after response removal,
    # but subtract it anyway.
    # ---------------------------------------------------------
    medianR = np.nanmedian(R)
    madR = np.nanmedian(np.abs(R - medianR))

    # Convert MAD to approximately sigma-equivalent scale.
    sigmaRobust = 1.4826 * madR

    if sigmaRobust <= 0 or not np.isfinite(sigmaRobust):
        #raise ValueError("Could not estimate a valid robust background scale.")
        return []

    if thresholdHigh is None:
        thresholdHigh = medianR + thresholdHighFactor * sigmaRobust

    if thresholdLow is None:
        thresholdLow = medianR + thresholdLowFactor * sigmaRobust

    nQuiet = max(1, int(round(quietSec * fs)))
    nPad = int(round(padSec * fs))

    refinedEvents = []

    # ---------------------------------------------------------
    # 4. Search only inside broad raw-count events
    # ---------------------------------------------------------
    for broadStart, broadEnd in eventsPass:
        
        broadStartProc = int(broadStart)
        broadEndProc = int(broadEnd)

        # Clip safely
        broadStartProc = max(0, broadStartProc)
        broadEndProc = min(nSamples - 1, broadEndProc)

        if broadEndProc <= broadStartProc:
            continue

        inEvent = False
        startIndex = None
        quietCount = 0

        for i in range(broadStartProc, broadEndProc + 1):

            if not inEvent:

                if R[i] >= thresholdHigh:
                    inEvent = True
                    startIndex = i
                    quietCount = 0

            else:

                if R[i] < thresholdLow:
                    quietCount += 1

                    if quietCount >= nQuiet:

                        endIndex = i - nQuiet + 1

                        # Apply small guard interval
                        startPadded = max(broadStartProc,startIndex - nPad)

                        endPadded = min(broadEndProc,endIndex + nPad)

                        refinedEvents.append((startPadded, endPadded))

                        inEvent = False
                        startIndex = None
                        quietCount = 0

                else:
                    quietCount = 0

        # If spike remains active to the end of the broad raw event
        if inEvent:

            startPadded = max(broadStartProc,startIndex - nPad)

            endPadded = broadEndProc

            refinedEvents.append((startPadded, endPadded))
            
        # Padding can cause neighbouring G2 intervals to overlap.
        # In that case, treat them as one continuous contaminated
        # interval.
        # ---------------------------------------------------------
        if len(refinedEvents) > 1:

            mergedEvents = [refinedEvents[0]]

            for start, end in refinedEvents[1:]:

                prevStart, prevEnd = mergedEvents[-1]

                if start <= prevEnd:
                    mergedEvents[-1] = (prevStart,max(prevEnd, end))
                else:
                    mergedEvents.append((start, end))

            refinedEvents = mergedEvents
            
    return refinedEvents
    
def makeAccelerationStepTemplate(inventory,trace,templateSec=180.0,preSec=30.0,simulationSec=600.0):
    """
    Generate the theoretical raw-channel waveform produced by a
    step in ground acceleration, using the response stored in the
    StationXML inventory.

    The absolute amplitude of the synthetic step is arbitrary because
    the resulting template is normalized before template matching.

    Parameters
    ----------
    inventory : obspy.Inventory
        Station inventory containing the response of `trace`.

    trace : obspy.Trace
        A raw trace from the channel for which the template is required.
        Its network, station, location, channel, sampling rate and time
        are used to select the appropriate instrument response.

    templateSec : float
        Total duration of the returned template.

    preSec : float
        Amount of template time preceding the acceleration-step onset.

    simulationSec : float
        Length of the internal synthetic time series used for the
        frequency-domain response simulation. This should be much
        longer than templateSec so that the artificial end of the
        acceleration step does not affect the extracted template.

    Returns
    -------
    template : np.ndarray
        Normalized theoretical raw glitch waveform.

    tTemplate : np.ndarray
        Time relative to the acceleration-step onset. Therefore t=0
        corresponds to the theoretical glitch onset.
    """

    import numpy as np

    # ---------------------------------------------------------
    # 1. Sampling information
    # ---------------------------------------------------------
    fs = float(trace.stats.sampling_rate)

    nSimulation = int(round(simulationSec * fs))

    # Prefer an even number of samples for rFFT.
    if nSimulation % 2 != 0:
        nSimulation += 1

    # Put the acceleration step well away from the beginning.
    onsetSec = simulationSec / 3.0
    onsetIndex = int(round(onsetSec * fs))

    # ---------------------------------------------------------
    # 2. Synthetic unit step in acceleration
    # ---------------------------------------------------------
    acceleration = np.zeros(nSimulation, dtype=np.float64)

    acceleration[onsetIndex:] = 1.0

    # ---------------------------------------------------------
    # 3. Frequency axis
    # ---------------------------------------------------------
    frequencies = np.fft.rfftfreq(nSimulation,d=1.0/fs)

    accelerationFFT = np.fft.rfft(acceleration)

    # ---------------------------------------------------------
    # 4. Obtain the actual response for this channel
    #
    # output="ACC" gives the transfer function appropriate for
    # acceleration -> recorded counts.
    # ---------------------------------------------------------
    response = inventory.get_response(trace.id,trace.stats.starttime)

    responseFFT = response.get_evalresp_response_for_frequencies(frequencies,output="ACC")

    responseFFT = np.asarray(responseFFT,dtype=np.complex128)

    # DC is not useful for a broadband seismometer and may contain
    # a singular/undefined value depending on the response description.
    responseFFT[0] = 0.0

    # ---------------------------------------------------------
    # 5. Forward simulation:
    #
    # acceleration -> instrument response -> raw channel
    # ---------------------------------------------------------
    syntheticRawFFT = accelerationFFT * responseFFT

    syntheticRaw = np.fft.irfft(syntheticRawFFT,n=nSimulation)

    # ---------------------------------------------------------
    # 6. Extract a window around the step onset
    # ---------------------------------------------------------
    nPre = int(round(preSec * fs))
    nTemplate = int(round(templateSec * fs))

    startIndex = onsetIndex - nPre
    endIndex = startIndex + nTemplate

    if startIndex < 0 or endIndex > nSimulation:
        raise ValueError("simulationSec is too short for the requested template.")

    template = syntheticRaw[startIndex:endIndex].copy()

    tTemplate = (np.arange(len(template)) / fs- preSec)

    # ---------------------------------------------------------
    # 7. Remove any tiny numerical baseline
    # ---------------------------------------------------------
    preMask = tTemplate < 0

    if np.any(preMask):
        template -= np.mean(template[preMask])

    # ---------------------------------------------------------
    # 8. Normalize shape
    #
    # Sign is intentionally NOT forced. When matching real data
    # later we can allow both +template and -template.
    # ---------------------------------------------------------
    norm = np.max(np.abs(template))

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Could not generate a valid acceleration-step template.")

    template /= norm

    return template, tTemplate