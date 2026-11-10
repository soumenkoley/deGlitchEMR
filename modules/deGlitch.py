import numpy as np
import scipy.io as sio
from scipy.io import savemat,loadmat
import scipy.signal as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math as mt
import os.path as op
import obspy
import statistics as st
from obspy.core import read
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy import UTCDateTime
from obspy import read_inventory
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import cdist
from scipy.signal import periodogram
from scipy import signal
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.optimize import minimize
import pickle
import datetime
import pywt
from fractions import Fraction
import pywt
from scipy.signal import resample_poly
from scipy.signal import fftconvolve
from scipy.linalg import solve
from scipy.linalg import lstsq
from obspy.clients.filesystem.sds import Client
from obspy.clients.fdsn import Client as ClientFDSN
#from modules import deGlitchUtil

def findValueInList(targetValue, nestedList):
    # Enumerate gives both the list index (idx) and the inner list itself
    for idx, innerList in enumerate(nestedList):
        if targetValue in innerList:
            return True, idx  # Found it! Return True and the index
            
    return False, None  # Not found anywhere

def makeAccelerationStepTemplate(inventory,network,station,location,channel,fs,responseTime,templateSec=180.0,
                                 preSec=30.0,simulationSec=600.0,normalize=True):

    # ---------------------------------------------------------
    # 1. Sampling
    # ---------------------------------------------------------
    fs = float(fs)

    nSimulation = int(round(simulationSec * fs))

    if nSimulation % 2 != 0:
        nSimulation += 1

    # Put onset well inside the synthetic record
    onsetSec = simulationSec / 3.0
    onsetIndex = int(round(onsetSec * fs))

    # ---------------------------------------------------------
    # 2. Unit step in acceleration
    # ---------------------------------------------------------
    acceleration = np.zeros(nSimulation,dtype=np.float64)

    acceleration[onsetIndex:] = 1.0

    # ---------------------------------------------------------
    # 3. Frequency axis
    # ---------------------------------------------------------
    frequencies = np.fft.rfftfreq(nSimulation,d=1.0/fs)

    accelerationFFT = np.fft.rfft(acceleration)

    # ---------------------------------------------------------
    # 4. Get response from inventory
    # ---------------------------------------------------------
    seedID = (f"{network}."f"{station}."f"{location}."f"{channel}")

    response = inventory.get_response(seedID,responseTime)

    responseFFT = (response.get_evalresp_response_for_frequencies(frequencies,output="ACC"))

    responseFFT = np.asarray(responseFFT,dtype=np.complex128)

    # Ignore DC
    responseFFT[0] = 0.0

    # ---------------------------------------------------------
    # 5. Forward simulation
    #
    # acceleration -> instrument -> raw counts
    # ---------------------------------------------------------
    syntheticRawFFT = (accelerationFFT * responseFFT)

    syntheticRaw = np.fft.irfft(syntheticRawFFT,n=nSimulation)

    # ---------------------------------------------------------
    # 6. Extract template around onset
    # ---------------------------------------------------------
    nPre = int(round(preSec * fs))
    nTemplate = int(round(templateSec * fs))

    startIndex = onsetIndex - nPre
    endIndex = startIndex + nTemplate

    if startIndex < 0 or endIndex > nSimulation:
        raise ValueError("simulationSec is too short for ""the requested template.")

    template = syntheticRaw[startIndex:endIndex].copy()

    tTemplate = (np.arange(len(template))/fs-preSec)

    # ---------------------------------------------------------
    # 7. Remove pre-event baseline
    # ---------------------------------------------------------
    preMask = tTemplate < 0

    if np.any(preMask):template -= np.mean(template[preMask])

    # ---------------------------------------------------------
    # 8. Normalize
    # ---------------------------------------------------------
    norm = np.max(np.abs(template))

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Could not generate a valid template.")

    #template /= norm
    if normalize:
        scale = np.max(np.abs(template))
        template = template/scale
    else:
        scale = 1.0

    return template, tTemplate,scale

def makeDispStepTemplate(inventory,network,station,location,channel,fs,responseTime,templateSec=180.0,
                                 preSec=30.0,simulationSec=600.0,normalize=True):


    # ---------------------------------------------------------
    # 1. Sampling
    # ---------------------------------------------------------
    fs = float(fs)

    nSimulation = int(round(simulationSec * fs))

    if nSimulation % 2 != 0:
        nSimulation += 1

    # Put onset well inside the synthetic record
    onsetSec = simulationSec / 3.0
    onsetIndex = int(round(onsetSec * fs))

    # ---------------------------------------------------------
    # 2. Unit step in acceleration
    # ---------------------------------------------------------
    acceleration = np.zeros(nSimulation,dtype=np.float64)

    acceleration[onsetIndex:] = 1.0

    # ---------------------------------------------------------
    # 3. Frequency axis
    # ---------------------------------------------------------
    frequencies = np.fft.rfftfreq(nSimulation,d=1.0/fs)

    accelerationFFT = np.fft.rfft(acceleration)

    # ---------------------------------------------------------
    # 4. Get response from inventory
    # ---------------------------------------------------------
    seedID = (f"{network}."f"{station}."f"{location}."f"{channel}")

    response = inventory.get_response(seedID,responseTime)

    responseFFT = (response.get_evalresp_response_for_frequencies(frequencies,output="DISP"))

    responseFFT = np.asarray(responseFFT,dtype=np.complex128)

    # Ignore DC
    responseFFT[0] = 0.0

    # ---------------------------------------------------------
    # 5. Forward simulation
    #
    # acceleration -> instrument -> raw counts
    # ---------------------------------------------------------
    syntheticRawFFT = (accelerationFFT * responseFFT)

    syntheticRaw = np.fft.irfft(syntheticRawFFT,n=nSimulation)

    # ---------------------------------------------------------
    # 6. Extract template around onset
    # ---------------------------------------------------------
    nPre = int(round(preSec * fs))
    nTemplate = int(round(templateSec * fs))

    startIndex = onsetIndex - nPre
    endIndex = startIndex + nTemplate

    if startIndex < 0 or endIndex > nSimulation:
        raise ValueError("simulationSec is too short for ""the requested template.")

    template = syntheticRaw[startIndex:endIndex].copy()

    tTemplate = (np.arange(len(template))/fs-preSec)

    # ---------------------------------------------------------
    # 7. Remove pre-event baseline
    # ---------------------------------------------------------
    preMask = tTemplate < 0

    if np.any(preMask):template -= np.mean(template[preMask])

    # ---------------------------------------------------------
    # 8. Normalize
    # ---------------------------------------------------------
    norm = np.max(np.abs(template))

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Could not generate a valid template.")

    #template /= norm
    if normalize:
        scale = np.max(np.abs(template))
        template = template/scale
    else:
        scale = 1.0

    return template, tTemplate,scale

def getDispTemplateResolution(template,tTemplate,corrThreshold=0.5,energyFraction=0.9999):

    """
    Estimate temporal resolving power from template autocorrelation.

    Returns the first positive lag where |autocorrelation|
    falls below corrThreshold.
    """

    g = np.asarray(template,dtype=np.float64).copy()
    tg = np.asarray(tTemplate,dtype=np.float64)

    fs = (1.0/ np.median(np.diff(tg)))
    onsetTemplate = int(np.argmin(np.abs(tg)))

    if onsetTemplate > 0:
        g -= np.mean(g[:onsetTemplate])
    else:
        g -= g[0]

    # Compact energetic part only
    energy = g**2

    totalEnergy = np.sum(energy)

    if totalEnergy <= 0:
        raise ValueError("Invalid displacement template.")

    cdf = (np.cumsum(energy)/ totalEnergy)
    tail = (1.0 - energyFraction) / 2.0
    i1 = int(np.searchsorted(cdf,tail))
    i2 = int(np.searchsorted(cdf,1.0 - tail))

    gCompact = g[i1:i2 + 1]
    autocorr = sp.correlate(gCompact,gCompact,mode="full")

    mid = len(gCompact) - 1

    autocorrPositive = (autocorr[mid:]/autocorr[mid])
    resolutionSamples = None

    for lag in range(1,len(autocorrPositive)):
        if (abs(autocorrPositive[lag])<= corrThreshold):
            resolutionSamples = lag
            break

    if resolutionSamples is None:
        resolutionSamples = max(1,len(gCompact) // 4)

    return {"samples":int(resolutionSamples),"seconds":float(resolutionSamples/ fs),
            "autocorrelation":autocorrPositive,"fs":fs}

def detectPhysicsG1VariancePursuit(rawStream,template,tTemplate,hourStart,extractLen=3600.0,overlapBuffer=60.0,
                                   lowpass=0.5,maxEvents=50,shortlistSize=20,minIncrementalVR=None,
                                   targetVR=None,minSeparationSec=0.0,pinvRcond=1.0e-10):
    """
    Forward selection of acceleration-step responses based on
    maximum incremental variance reduction.

    At every iteration:

        1. Search ALL possible onset samples using FFT correlation.
        2. Take a small shortlist of the best candidates.
        3. For each shortlisted candidate, calculate the exact
           incremental SSE reduction after accounting for all
           previously selected templates.
        4. Add the best candidate.
        5. Jointly refit amplitudes of ALL selected events.
        6. Repeat.

    No initial number of glitches or onset times is supplied.

    The returned 'selected_coefficients' has the same length as
    the processed trace and is zero except at selected onset times.
    """

    # =========================================================
    # 1. Prepare raw trace
    # =========================================================

    if rawStream is None or len(rawStream) == 0:
        return None

    st = rawStream.copy()
    st.merge(method=1, fill_value=0)

    if len(st) != 1:
        raise ValueError("Expected one merged raw trace.")

    tr = st[0]

    fs = float(tr.stats.sampling_rate)

    startTime = hourStart - overlapBuffer
    endTime = hourStart + extractLen

    tr.trim(starttime=startTime,endtime=endTime,nearest_sample=False,pad=False)

    nExpected = int(round((overlapBuffer + extractLen) * fs))

    if len(tr.data) < nExpected:
        print(f"{tr.id}: expected {nExpected} samples, "f"found {len(tr.data)}.")
        return None

    x = np.asarray(tr.data[:nExpected],dtype=np.float64)
    nSamples = len(x)

    # =========================================================
    # 2. Prepare template
    # =========================================================

    g = np.asarray(template,dtype=np.float64).copy()
    tg = np.asarray(tTemplate,dtype=np.float64)

    if len(g) != len(tg):
        raise ValueError("template and tTemplate must have same length.")

    fsTemplate = (1.0/np.median(np.diff(tg)))

    if not np.isclose(fsTemplate,fs,rtol=1.0e-3):
        raise ValueError(f"Template fs={fsTemplate:.6f} Hz, "f"data fs={fs:.6f} Hz.")

    onsetTemplate = int(np.argmin(np.abs(tg)))

    # Remove pre-event baseline from template
    if onsetTemplate > 0:
        g -= np.mean(g[:onsetTemplate])
    else:
        g -= g[0]

    # =========================================================
    # 3. Apply identical low-pass
    # =========================================================

    if lowpass is not None:
        sos = sp.butter(4,lowpass,btype="lowpass",fs=fs,output="sos")
        xFit = sp.sosfiltfilt(sos,x)
        gFit = sp.sosfiltfilt(sos,g)
    else:
        xFit = x.copy()
        gFit = g.copy()

    # =========================================================
    # 4. Normalize template internally
    # =========================================================

    gNorm = np.linalg.norm(gFit)
    if(not np.isfinite(gNorm) or gNorm == 0):
        raise ValueError("Invalid acceleration-step template.")
    gWork = (gFit/gNorm)
    nTemplate = len(gWork)

    # =========================================================
    # 5. Constant + linear background projection
    # =========================================================

    tBaseline = np.linspace(-1.0,1.0,nSamples)
    B = np.column_stack((np.ones(nSamples),tBaseline))

    BtBInv = np.linalg.inv(B.T @ B)
    def projectOutBaseline(v):
        betaBackground = (BtBInv @ (B.T @ v))
        return (v- B @ betaBackground)

    xProjected = projectOutBaseline(xFit)
    sseBaseline = np.dot(xProjected,xProjected)

    if sseBaseline <= 0:
        raise RuntimeError("Zero baseline-projected variance.")
    # =========================================================
    # 6. Forward/adjoint operators
    # =========================================================

    def forward(s):
        full = sp.fftconvolve(s,gWork,mode="full")
        return full[onsetTemplate:onsetTemplate + nSamples]

    def adjointWithKernel(r,kernel):
        q = np.zeros(nSamples + len(kernel) - 1,dtype=np.float64)
        q[onsetTemplate:onsetTemplate + nSamples] = r
        return sp.fftconvolve(q,kernel[::-1],mode="valid")

    def adjoint(r):
        return adjointWithKernel(r,gWork)

    # =========================================================
    # 7. Construct one shifted template explicitly
    # =========================================================

    def makeColumn(onsetIndex):

        column = np.zeros(nSamples,dtype=np.float64)
        templateStart = (onsetIndex- onsetTemplate)
        dataStart = max(0,templateStart)
        dataEnd = min(nSamples,templateStart + nTemplate)
        if dataEnd <= dataStart:
            return column

        templateLocalStart = (dataStart - templateStart)
        templateLocalEnd = (templateLocalStart+ dataEnd - dataStart)
        column[dataStart:dataEnd] = gWork[templateLocalStart:templateLocalEnd]
        return column

    # =========================================================
    # 8. Precompute projected norm of every possible atom
    #
    # This gives us a normalized first-pass score without
    # constructing a 146000-column dictionary.
    # =========================================================

    onesData = np.ones(nSamples,dtype=np.float64)

    rawAtomNormSq = (adjointWithKernel(onesData,gWork**2))

    # <g_i, baseline column>
    atomBaseline0 = adjoint(B[:, 0])
    atomBaseline1 = adjoint(B[:, 1])
    atomBaselineInner = np.column_stack((atomBaseline0,atomBaseline1))

    baselineCorrection = np.einsum("ij,jk,ik->i",atomBaselineInner,BtBInv,atomBaselineInner)

    projectedAtomNormSq = (rawAtomNormSq- baselineCorrection)

    projectedAtomNormSq = np.maximum(projectedAtomNormSq,1.0e-20)
    # =========================================================
    # 9. Storage
    # =========================================================

    maxEvents = int(maxEvents)

    # Only maxEvents columns are ever stored.
    # Example: 146400 x 50 ~= 59 MB in float64.
    selectedMatrix = np.zeros((nSamples, maxEvents),dtype=np.float64)
    gram = np.zeros((maxEvents, maxEvents),dtype=np.float64)
    rhs = np.zeros(maxEvents,dtype=np.float64)
    selectedIndices = []
    vrHistory = []
    incrementalVRHistory = []
    sseHistory = []

    residualProjected = (xProjected.copy())
    ssePrevious = (sseBaseline)

    beta = np.array([],dtype=np.float64)
    gramInverse = None
    nSeparation = int(round(minSeparationSec * fs))
    # =========================================================
    # 10. Forward variance-reduction pursuit
    # =========================================================

    for iteration in range(maxEvents):

        K = len(selectedIndices)

        # -----------------------------------------------------
        # Search all possible onset samples simultaneously
        # -----------------------------------------------------

        correlation = adjoint(residualProjected)
        # First-pass approximate SSE reduction
        searchScore = (correlation**2/projectedAtomNormSq)
        # Do not select an existing event again
        for ind in selectedIndices:

            i1 = max(0,ind - nSeparation)
            i2 = min(nSamples,ind + nSeparation + 1)

            searchScore[
                i1:i2
            ] = -np.inf

        finiteMask = np.isfinite(searchScore)

        nAvailable = np.sum(finiteMask)

        if nAvailable == 0:
            break

        nShort = min(shortlistSize,nAvailable)
        # -----------------------------------------------------
        # Take only the strongest candidates for exact testing
        # -----------------------------------------------------

        validIndices = np.flatnonzero(finiteMask)
        validScores = searchScore[validIndices]
        if nShort < len(validIndices):
            localTop = np.argpartition(validScores,-nShort)[-nShort:]

            candidateIndices = (validIndices[localTop])
        else:
            candidateIndices = (validIndices)

        # Sort strongest approximate candidate first
        candidateIndices = candidateIndices[
            np.argsort(searchScore[candidateIndices])[::-1]]
        # -----------------------------------------------------
        # Exact incremental SSE reduction for shortlist
        #
        # Candidate is orthogonalized against all previously
        # selected physical responses.
        # -----------------------------------------------------

        bestIndex = None
        bestColumnProjected = None
        bestDeltaSSE = -np.inf

        for candidateIndex in candidateIndices:

            candidateColumn = makeColumn(int(candidateIndex))
            candidateColumnProjected = (projectOutBaseline(candidateColumn))

            atomNorm = np.dot(candidateColumnProjected,candidateColumnProjected)
            if atomNorm <= 0:
                continue

            numerator = np.dot(candidateColumnProjected,residualProjected)
            if K == 0:
                denominator = (atomNorm)
            else:
                cross = (selectedMatrix[:,:K].T @ candidateColumnProjected)
                denominator = (atomNorm - cross @ gramInverse @ cross)
            # Numerical degeneracy:
            # candidate already represented by existing atoms.
            if (not np.isfinite(denominator)or denominator<= 1.0e-12 * atomNorm):
                continue

            deltaSSE = (numerator**2/denominator)

            if (deltaSSE> bestDeltaSSE):
                bestDeltaSSE = (deltaSSE)
                bestIndex = int(candidateIndex)
                bestColumnProjected = (candidateColumnProjected)

        if bestIndex is None:
            break

        predictedIncrementalVR = (bestDeltaSSE/sseBaseline)

        # Optional stopping rule
        if (minIncrementalVR is not None and predictedIncrementalVR< minIncrementalVR):
            break

        # -----------------------------------------------------
        # Add best candidate
        # -----------------------------------------------------

        selectedIndices.append(bestIndex)
        selectedMatrix[:, K] = (bestColumnProjected)
        # Update Gram matrix
        if K > 0:

            crossNew = (selectedMatrix[:, :K].T @ bestColumnProjected)
            gram[:K,K] = crossNew
            gram[K,:K] = crossNew
        
        gram[K,K] = np.dot(bestColumnProjected,bestColumnProjected)

        rhs[K] = np.dot(bestColumnProjected,xProjected)
        Knew = K + 1

        # -----------------------------------------------------
        # Jointly refit ALL selected amplitudes
        # -----------------------------------------------------

        gramNow = gram[:Knew,:Knew]

        rhsNow = rhs[:Knew]
        gramInverse = np.linalg.pinv(gramNow,rcond=pinvRcond)

        beta = (gramInverse @ rhsNow)

        modelProjected = (selectedMatrix[:, :Knew] @ beta)
        residualProjected = (xProjected - modelProjected)

        sseNow = np.dot(residualProjected,residualProjected)
        vrNow = (1.0 - sseNow/sseBaseline)

        incrementalVR = (ssePrevious - sseNow) / sseBaseline

        vrHistory.append(vrNow)
        incrementalVRHistory.append(incrementalVR)
        sseHistory.append(sseNow)
        ssePrevious = (sseNow)

        # Optional total-VR stopping
        if (targetVR is not None and vrNow >= targetVR):
            break

    # =========================================================
    # 11. Build final sparse coefficient sequence
    # =========================================================

    selectedIndices = np.asarray(selectedIndices,dtype=int)

    selectedCoefficientsInternal = (np.zeros(nSamples,dtype=np.float64))

    if len(selectedIndices) > 0:

        selectedCoefficientsInternal[selectedIndices] = beta

    # Convert back to scaling of ORIGINAL supplied template
    selectedCoefficients = (selectedCoefficientsInternal/ gNorm)

    # =========================================================
    # 12. Final physical reconstruction
    # =========================================================

    glitchModel = forward(selectedCoefficientsInternal)
    remaining = (xFit- glitchModel)
    betaBackground = (BtBInv @ (B.T @ remaining))

    background = ( B @ betaBackground)

    fullModel = (glitchModel + background)
    residual = (xFit- fullModel)

    # =========================================================
    # 13. Timing
    # =========================================================

    selectedTimesTrace = (selectedIndices.astype(float) / fs)
    selectedTimesHour = (selectedTimesTrace - overlapBuffer)

    return {"seed_id":tr.id,"fs":fs,"startTime":startTime,"raw":x,"raw_filtered":xFit,
        # Main output
        "selected_coefficients":
            selectedCoefficients,
        "selected_indices":
            selectedIndices,
        "selected_times_trace":
            selectedTimesTrace,
        "selected_times_hour":
            selectedTimesHour,
        "selected_amplitudes":selectedCoefficients[selectedIndices],
        # Model
        "glitch_model":glitchModel,
        "background":background,
        "full_model":fullModel,
        "residual":residual,
        # Most important diagnostics
        "variance_reduction":np.asarray(vrHistory),
        "incremental_variance_reduction":np.asarray(incrementalVRHistory),
        "sse_history":np.asarray(sseHistory),
        "n_selected":len(selectedIndices)}

def backwardPrunePhysicsCoefficients(res,template,tTemplate,lowpass=0.5,rcond=1.0e-10):
    """
    Backward pruning of step-response coefficients returned by
    detectPhysicsG1VariancePursuit().

    The onset times are kept fixed.

    At each iteration:
        1. Jointly refit all currently retained amplitudes.
        2. Remove each coefficient individually.
        3. Refit all remaining amplitudes.
        4. Calculate the loss in variance reduction.
        5. Permanently remove the coefficient causing the
           smallest loss.
        6. Repeat.

    Returns the complete pruning path Kmax -> ... -> 1 -> 0.

    Parameters
    ----------
    res : dict
        Output of detectPhysicsG1VariancePursuit().
    template, tTemplate :
        Same acceleration-step template used in the pursuit.
    lowpass : float or None
        Must be the same low-pass used in the pursuit.
    rcond : float
        Conditioning parameter for small least-squares systems.
    """

    # =========================================================
    # 1. Data and selected onset locations
    # =========================================================

    xFit = np.asarray(res["raw_filtered"],dtype=np.float64)

    selectedIndices = np.asarray(res["selected_indices"],dtype=int)

    fs = float(res["fs"])

    nSamples = len(xFit)
    nInitial = len(selectedIndices)

    if nInitial == 0:
        raise ValueError("No coefficients available for backward pruning.")

    # =========================================================
    # 2. Prepare template exactly as in forward pursuit
    # =========================================================

    g = np.asarray(template,dtype=np.float64).copy()

    tg = np.asarray(tTemplate,dtype=np.float64)

    if len(g) != len(tg):
        raise ValueError("template and tTemplate must have same length.")

    fsTemplate = (1.0/np.median(np.diff(tg)))

    if not np.isclose(fsTemplate,fs,rtol=1.0e-3):

        raise ValueError(f"Template fs={fsTemplate:.6f} Hz, "f"data fs={fs:.6f} Hz.")

    onsetTemplate = int(np.argmin(np.abs(tg)))

    # Same template baseline removal
    if onsetTemplate > 0:
        g -= np.mean(g[:onsetTemplate])
    else:
        g -= g[0]

    # Same low-pass
    if lowpass is not None:
        sos = sp.butter(4,lowpass,btype="lowpass",fs=fs,output="sos")
        gFit = sp.sosfiltfilt(sos,g)
    else:
        gFit = g.copy()
    nTemplate = len(gFit)

    # =========================================================
    # 3. Constant + linear background projection
    # =========================================================

    tBaseline = np.linspace(-1.0,1.0,nSamples)
    B = np.column_stack((np.ones(nSamples),tBaseline))
    BtBInv = np.linalg.inv(B.T @ B)

    def projectOutBaseline(v):
        return (v-B@(BtBInv@(B.T@v)))

    xProjected = projectOutBaseline(xFit)

    sseBaseline = np.dot(xProjected,xProjected)

    # =========================================================
    # 4. Build physical template matrix ONCE
    #
    # One column per onset selected by forward pursuit.
    # =========================================================

    G = np.zeros((nSamples, nInitial),dtype=np.float64)

    for j, onsetIndex in enumerate(selectedIndices):
        templateStart = (onsetIndex-onsetTemplate)
        dataStart = max(0,templateStart)

        dataEnd = min(nSamples,templateStart + nTemplate)

        if dataEnd <= dataStart:
            continue

        templateStartLocal = (dataStart-templateStart)
        templateEndLocal = (templateStartLocal+dataEnd-dataStart)
        G[dataStart:dataEnd,j] = gFit[templateStartLocal:templateEndLocal]

    # Project constant and trend out of every column simultaneously
    GProjected = (G- B @(BtBInv@ (B.T @ G)))

    # =========================================================
    # 5. Reduce whole problem to small K x K system
    # =========================================================

    gramFull = (GProjected.T@ GProjected)

    rhsFull = (GProjected.T@ xProjected)

    # =========================================================
    # Helper: fit any subset
    # =========================================================

    def fitSubset(ids):
        ids = np.asarray(ids,dtype=int)
        if len(ids) == 0:

            return {"beta":np.array([]),"sse":sseBaseline,"vr":0.0}

        gram = gramFull[np.ix_(ids, ids)]
        rhs = rhsFull[ids]

        # Tiny system, so robust least squares is cheap.
        beta = np.linalg.lstsq(gram,rhs,rcond=rcond)[0]

        # SSE = x'x - beta' G'x
        sse = (sseBaseline- np.dot(beta,rhs))
        # Numerical roundoff protection
        sse = max(float(sse),0.0)

        vr = (1.0- sse / sseBaseline)

        return {"beta":beta,"sse":sse,"vr":vr}

    # =========================================================
    # 6. Backward elimination
    # =========================================================

    # These IDs refer to positions in the ORIGINAL list of
    # selectedIndices.
    currentIds = np.arange(nInitial,dtype=int)

    states = []

    removalOrder = []
    removalTimes = []
    removedDeltaVR = []
    kBeforeRemoval = []

    while len(currentIds) > 0:
        K = len(currentIds)

        # -----------------------------------------------------
        # Full fit with current K coefficients
        # -----------------------------------------------------

        fitCurrent = fitSubset(currentIds)
        vrCurrent = (fitCurrent["vr"])
        sseCurrent = (fitCurrent["sse"])
        # -----------------------------------------------------
        # Remove every coefficient once and refit everything
        # -----------------------------------------------------
        deltaVR = np.zeros(K,dtype=np.float64)
        vrWithout = np.zeros(K,dtype=np.float64)

        for j in range(K):
            trialIds = np.delete(currentIds,j)
            fitMinus = fitSubset(trialIds)
            vrWithout[j] = (fitMinus["vr"])
            deltaVR[j] = (vrCurrent-fitMinus["vr"])

        # Protect against tiny negative numerical values
        deltaVR = np.maximum(deltaVR,0.0)
        # -----------------------------------------------------
        # Save current state BEFORE removing anything
        # -----------------------------------------------------

        states.append({"K":K,"candidate_ids":currentIds.copy(),"indices":selectedIndices[currentIds].copy(),
                       "times_trace":selectedIndices[currentIds].astype(float) / fs,"amplitudes":fitCurrent["beta"].copy(),
                       "variance_reduction":vrCurrent,"sse":sseCurrent,"delta_vr":deltaVR.copy(),"vr_without":vrWithout.copy()})

        # -----------------------------------------------------
        # Least indispensable response
        # -----------------------------------------------------

        removeLocal = int(np.argmin(deltaVR))
        removeCandidateId = int(currentIds[removeLocal])

        removalOrder.append(removeCandidateId)
        removalTimes.append(selectedIndices[removeCandidateId]/fs)
        removedDeltaVR.append(deltaVR[removeLocal])
        kBeforeRemoval.append(K)

        # Permanently remove it
        currentIds = np.delete(currentIds,removeLocal)

    # =========================================================
    # 7. Add K=0 state
    # =========================================================

    states.append({"K":0,"candidate_ids":np.array([], dtype=int),"indices":np.array([], dtype=int),
                   "times_trace":np.array([], dtype=float),"amplitudes":np.array([], dtype=float),
                   "variance_reduction":0.0,"sse":sseBaseline,"delta_vr":np.array([], dtype=float),
                   "vr_without":np.array([], dtype=float)})

    # =========================================================
    # 8. Convenient pruning curves
    # =========================================================

    pruningK = np.asarray([state["K"] for state in states],dtype=int)

    pruningVR = np.asarray([state["variance_reduction"] for state in states],dtype=float)

    return {"states":states,"initial_indices":selectedIndices,"initial_times_trace":selectedIndices.astype(float)/fs,
            "n_initial":nInitial,"removal_order_candidate_id":np.asarray(removalOrder,dtype=int),
            "removal_order_index":selectedIndices[np.asarray(removalOrder,dtype=int)],
            "removal_order_time":np.asarray(removalTimes,dtype=float),"removed_delta_vr":np.asarray(removedDeltaVR,dtype=float),
            "k_before_removal":np.asarray(kBeforeRemoval,dtype=int),"pruning_k":pruningK,"pruning_vr":pruningVR,
            "baseline_sse":sseBaseline,"fs":fs}

def selectPrunedState(prune,deltaVRThreshold=1e-3):
    states = prune["states"]
    for i, state in enumerate(states[:-1]):
        minDeltaVR = np.min(state["delta_vr"])
        if minDeltaVR >= deltaVRThreshold:
            return state
    return states[-1]

def detectPhysicsG1HorizontalWindows(streamAll,horizontalIndices,template,tTemplate,hourStart,
                                     extractLen=3600.0,overlapBuffer=60.0,lowpass=0.5,maxEvents=40,shortlistSize=20,
                                     highConfidenceVR=0.8,deltaVRThreshold=1.0e-3,preSec=30.0,postSec=180.0):
    """
    Production physics-based G1 detector.

    Runs independently on every available horizontal component,
    combines the retained acceleration-step onset indices, builds
    [-preSec, +postSec] G1 windows, and merges overlaps.

    Returned intervals are in processedStream coordinates:
        index 0 = hourStart - overlapBuffer

    Therefore NO old dataBuffA-dataBuffB index correction is needed.
    """

    componentResults = []
    retainedIndicesAll = []

    hasData = False
    fsUse = None
    nSamplesUse = None

    # ---------------------------------------------------------
    # Run component-wise physics G1
    # ---------------------------------------------------------

    for indNow in horizontalIndices:
        if indNow is None:
            continue
        if indNow < 0 or indNow >= len(streamAll):
            continue
        stNow = streamAll[indNow]
        if stNow is None or len(stNow) == 0:
            continue
        if len(stNow[0].data) == 0:
            continue
        hasData = True
        # -----------------------------------------------------
        # Forward variance pursuit
        # -----------------------------------------------------
        res = detectPhysicsG1VariancePursuit(stNow,template,tTemplate,hourStart=hourStart,extractLen=extractLen,
                                             overlapBuffer=overlapBuffer,lowpass=lowpass,maxEvents=maxEvents,
                                             shortlistSize=shortlistSize,minIncrementalVR=None,targetVR=None,
                                             minSeparationSec=0.0)
        if res is None:
            componentResults.append({"stream_index": indNow,"status": "failed","full_vr": np.nan,
                                     "retained_indices": np.array([], dtype=int)})
            continue

        fsUse = float(res["fs"])
        nSamplesUse = len(res["raw_filtered"])

        # -----------------------------------------------------
        # Full 40-event model VR
        # -----------------------------------------------------

        if len(res["variance_reduction"]) == 0:
            fullVR = 0.0
        else:
            fullVR = float(res["variance_reduction"][-1])

        # -----------------------------------------------------
        # High-confidence hour/component gate
        # -----------------------------------------------------

        if fullVR < highConfidenceVR:
            componentResults.append({"stream_index": indNow,"status": "low_vr","full_vr": fullVR,
                                     "n_candidates": len(res["selected_indices"]),"n_retained": 0,
                                     "retained_indices": np.array([], dtype=int),"pursuit": res})
            continue

        # -----------------------------------------------------
        # Backward pruning
        # -----------------------------------------------------
        prune = backwardPrunePhysicsCoefficients(res,template,tTemplate,lowpass=lowpass)
        stateUseful = selectPrunedState(prune,deltaVRThreshold=deltaVRThreshold)

        retainedIndices = np.asarray(stateUseful["indices"],dtype=int)

        retainedIndicesAll.extend(retainedIndices.tolist())

        componentResults.append({"stream_index": indNow,"status": "accepted","full_vr": fullVR,
                                 "retained_vr": float(stateUseful["variance_reduction"]),
                                 "n_candidates": len(res["selected_indices"]),"n_retained": len(retainedIndices),
                                 "retained_indices": retainedIndices,"retained_times_trace":retainedIndices.astype(float)/fsUse,
                                 "retained_amplitudes":stateUseful["amplitudes"].copy(),"pursuit": res,"prune": prune})

    # ---------------------------------------------------------
    # No usable horizontal data
    # ---------------------------------------------------------

    if not hasData:
        return {"has_data": False,"intervals": [],"retained_indices": np.array([], dtype=int),"component_results": componentResults}

    # ---------------------------------------------------------
    # Valid data, but no high-confidence G1 detections
    # ---------------------------------------------------------

    if len(retainedIndicesAll) == 0:
        return {"has_data": True,"intervals": [],"retained_indices": np.array([], dtype=int),"component_results": componentResults}

    # ---------------------------------------------------------
    # Union BH1/BH2 onset indices
    # ---------------------------------------------------------

    retainedIndicesAll = np.unique(np.asarray(retainedIndicesAll,dtype=int))

    # ---------------------------------------------------------
    # Generate [-30 s, +180 s] windows
    # ---------------------------------------------------------

    nPre = int(round(preSec * fsUse))
    nPost = int(round(postSec * fsUse))
    intervals = []
    for indNow in retainedIndicesAll:
        i1 = max(0,int(indNow - nPre))
        i2 = min(nSamplesUse - 1,int(indNow + nPost))
        intervals.append((i1, i2))
    # ---------------------------------------------------------
    # Merge overlapping G1 windows
    # ---------------------------------------------------------

    intervals.sort(key=lambda x: x[0])
    merged = []
    for i1, i2 in intervals:
        if len(merged) == 0:

            merged.append([i1, i2])

        elif i1 <= merged[-1][1] + 1:

            merged[-1][1] = max(merged[-1][1],i2)
        else:
            merged.append([i1, i2])

    merged = [(int(i1), int(i2)) for i1, i2 in merged]
    return {"has_data": True,
        # This is what G2 needs
        "intervals": merged,
        # Diagnostics
        "retained_indices":
            retainedIndicesAll,
        "retained_times_trace":
            retainedIndicesAll.astype(float) / fsUse,
        "component_results":
            componentResults
    }


def _robustSigma(x):
    """
    Robust Gaussian-equivalent scale using MAD.
    """

    x = np.asarray(x,dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return 1.0

    med = np.median(x)

    sigma = (1.4826*np.median(np.abs(x - med)))
    if (not np.isfinite(sigma) or sigma <= 1.0e-20):
        sigma = np.std(x)

    if (not np.isfinite(sigma) or sigma <= 1.0e-20):
        sigma = 1.0

    return float(sigma)


def _dilateMask1D(mask, nPad):
    """
    Binary dilation without requiring scipy.ndimage.
    """
    mask = np.asarray(mask,dtype=bool)
    nPad = int(max(0,nPad))
    if nPad == 0:
        return mask.copy()

    out = np.zeros_like(mask)
    inds = np.flatnonzero(mask)
    for ind in inds:

        i1 = max(0,ind - nPad)
        i2 = min(len(mask),ind + nPad + 1)
        out[i1:i2] = True

    return out


def _fillShortGaps(mask, maxGap):
    """
    Fill short False gaps between True samples.
    """

    mask = np.asarray(mask,dtype=bool).copy()
    maxGap = int(max(0,maxGap))
    if maxGap == 0:
        return mask

    trueInd = np.flatnonzero(mask)
    if len(trueInd) < 2:
        return mask

    for i1, i2 in zip(trueInd[:-1],trueInd[1:]):
        gap = (i2 - i1 - 1)
        if (gap > 0 and gap <= maxGap):
            mask[i1:i2 + 1] = True
    return mask

def _connectedRegions1D(mask):
    """
    Return inclusive (i1, i2) intervals of True regions.
    """

    mask = np.asarray(mask,dtype=bool)
    regions = []

    inRegion = False
    iStart = None

    for i, value in enumerate(mask):
        if (value and not inRegion):
            iStart = i
            inRegion = True

        elif (not value and inRegion):

            regions.append((iStart,i - 1))
            inRegion = False

    if inRegion:
        regions.append((iStart,len(mask) - 1))

    return regions


def _movingRMS(x, n):
    """
    Short moving RMS used only to make packet detection robust
    against zero crossings.
    """

    x = np.asarray(x,dtype=np.float64)
    n = int(max(1,n))
    n = min(n,len(x))
    kernel = (np.ones(n)/n)

    return np.sqrt(np.convolve(x**2,kernel,mode="same"))

def solveHPGuidedSWTWindow(xAll,xHPAll,interval,fs,fitPadSec=2.0,highSigma=5.0,lowSigma=2.5,
                           scoreSmoothSec=0.075,bridgeSec=0.075,maskPadSec=0.10,wavelet="sym4",swtLevel=5,
                           waveletSigma=3.0,coefficientPadSec=0.10,maxWaveletPadSec=0.50,
                           backgroundDegree=3,pinvRcond=1.0e-10):
    """
    HP-guided SWT deglitching for ONE G2 interval.

    The G2 interval can contain:
        - one isolated spike,
        - one long packet,
        - several isolated spikes,
        - several packets,
        - any mixture of these.

    Detection:
        high-pass multi-component robust score.

    Reconstruction:
        SWT detail coefficients only.

    The SWT is applied to the HIGH-PASS waveform.

    Therefore the reconstructed glitch model contains no DC/slow
    approximation term and can safely be subtracted from Stage-1
    raw counts without flattening the low-frequency seismic signal.
    """

    xAll = np.asarray(xAll,dtype=np.float64)
    xHPAll = np.asarray(xHPAll,dtype=np.float64)
    if xAll.shape != xHPAll.shape:
        raise ValueError("xAll and xHPAll must have identical shapes.")
    nComp, nSamples = xAll.shape

    # 1. G2 search interval
    i1 = max(0,int(interval[0]))
    i2 = min(nSamples - 1,int(interval[1]))
    if i2 < i1:
        return None

    nFitPad = int(round(fitPadSec * fs))
    fit1 = max(0,i1 - nFitPad)
    fit2 = min(nSamples - 1,i2 + nFitPad)
    globalSamples = np.arange(fit1,fit2 + 1,dtype=int)
    nLocal = len(globalSamples)
    Yraw = xAll[:,fit1:fit2 + 1].T
    Yhp = xHPAll[:,fit1:fit2 + 1].T

    # 2. Search-region and background masks
    g2Mask = (globalSamples >= i1) & (globalSamples <= i2)
    backgroundMask = ~g2Mask
    # If context outside G2 is insufficient, use the entire fit region.
    if np.sum(backgroundMask) < max(20,int(fs)):
        backgroundMask = np.ones(nLocal,dtype=bool)

    # 3. Robust multi-component high-pass score
    zHP = np.zeros_like(Yhp)
    hpSigma = np.zeros(nComp,dtype=np.float64)
    hpCenter = np.zeros(nComp,dtype=np.float64)
    for c in range(nComp):
        baseline = Yhp[backgroundMask,c]
        hpCenter[c] = np.median(baseline)
        hpSigma[c] = _robustSigma(baseline)
        zHP[:,c] = (Yhp[:,c] - hpCenter[c]) / hpSigma[c]

    scoreInstant = np.sqrt(np.mean(zHP**2,axis=1))
    # The short RMS keeps packets connected through oscillatory zero crossings.
    nSmooth = int(round(scoreSmoothSec * fs))
    scoreSmooth = _movingRMS(scoreInstant,nSmooth)
    transientScore = np.maximum(scoreInstant,scoreSmooth)

    # 4. Hysteretic packet detection: highSigma seed, lowSigma growth
    highMask = (transientScore >= highSigma) & g2Mask
    lowMask = (transientScore >= lowSigma) & g2Mask
    nBridge = int(round(bridgeSec * fs))
    lowMask = _fillShortGaps(lowMask,nBridge)
    glitchCoreMask = np.zeros(nLocal,dtype=bool)
    lowRegions = _connectedRegions1D(lowMask)

    for r1, r2 in lowRegions:
        if np.any(highMask[r1:r2 + 1]):
            glitchCoreMask[r1:r2 + 1] = True

    # 5. Determine packets. 'n_selected' counts packets, not SWT coefficients.
    packetRegionsLocal = _connectedRegions1D(glitchCoreMask)
    packetPeakLocal = []
    packetIntervalsGlobal = []
    packetPeakScores = []
    for p1, p2 in packetRegionsLocal:
        localPart = transientScore[p1:p2 + 1]
        iPeak = p1 + int(np.argmax(localPart))
        packetPeakLocal.append(iPeak)
        packetPeakScores.append(transientScore[iPeak])
        packetIntervalsGlobal.append((int(globalSamples[p1]),int(globalSamples[p2])))

    packetPeakLocal = np.asarray(packetPeakLocal,dtype=int)
    selectedIndices = np.asarray([globalSamples[i] for i in packetPeakLocal],dtype=int)
    packetPeakScores = np.asarray(packetPeakScores,dtype=np.float64)
    nPackets = len(selectedIndices)

    # 6. Expand packet mask slightly for wavelet reconstruction
    nMaskPad = int(round(maskPadSec * fs))
    glitchGuideMask = _dilateMask1D(glitchCoreMask,nMaskPad)

    # 7. No transient detected: retain the existing zero-correction output.
    if nPackets == 0:
        background = Yraw - Yhp
        return {
            "search_interval": (i1,i2),
            "fit_interval": (fit1,fit2),
            "n_selected": 0,
            "selected_indices": np.array([],dtype=int),
            "selected_samples_float": np.array([],dtype=float),
            "amplitudes": np.zeros((nComp,0)),
            "component_vr": np.zeros(nComp),
            "local_group_vr": 0.0,
            "transient_component_vr": np.zeros(nComp),
            "transient_group_vr": 0.0,
            "raw_component_vr": np.zeros(nComp),
            "raw_group_vr": 0.0,
            "model_local": np.zeros_like(Yraw),
            "cleaned_local": Yraw.copy(),
            "background": background,
            "glitch_mask_local": glitchCoreMask,
            "glitch_guide_mask_local": glitchGuideMask,
            "transient_score": transientScore,
            "packet_intervals": [],
            "packet_peak_scores": np.array([]),
            "wavelet_selected_counts": np.zeros(swtLevel,dtype=int),
            "n_wavelet_coefficients": 0,
            "event_correlations": np.array([]),
            "incremental_vr_history": np.array([])
        }

    # 8. Pad to a multiple of 2**swtLevel, as required by SWT
    block = 2**int(swtLevel)
    nExtra = (-nLocal) % block
    padLeft = nExtra // 2
    padRight = nExtra - padLeft
    YhpPad = np.zeros((nLocal + nExtra,nComp),dtype=np.float64)
    for c in range(nComp):
        if padLeft > 0 or padRight > 0:
            YhpPad[:,c] = np.pad(Yhp[:,c],(padLeft,padRight),mode="reflect")
        else:
            YhpPad[:,c] = Yhp[:,c]

    nPadTotal = YhpPad.shape[0]
    guidePad = np.zeros(nPadTotal,dtype=bool)
    guidePad[padLeft:padLeft + nLocal] = glitchGuideMask
    baselinePad = np.zeros(nPadTotal,dtype=bool)
    baselinePad[padLeft:padLeft + nLocal] = backgroundMask

    # 9. SWT decomposition for every component
    coeffsByComp = []
    cleanCoeffsByComp = []
    for c in range(nComp):
        coeffs = pywt.swt(YhpPad[:,c],wavelet,level=swtLevel,trim_approx=False,norm=False)
        coeffsByComp.append(coeffs)
        cleanCoeffsByComp.append([(cA.copy(),cD.copy()) for cA,cD in coeffs])

    waveletObject = pywt.Wavelet(wavelet)
    filterLength = waveletObject.dec_len
    selectedCounts = []

    # 10. Select abnormal SWT DETAIL coefficients (never remove approximation)
    for coeffIndex in range(swtLevel):
        # pywt.swt ordering: [(cA_J,cD_J),...,(cA_1,cD_1)]
        levelNumber = swtLevel - coeffIndex
        detail = np.vstack([coeffsByComp[c][coeffIndex][1] for c in range(nComp)])
        detailSigma = np.zeros(nComp,dtype=np.float64)
        for c in range(nComp):
            baseVals = detail[c,baselinePad]
            if len(baseVals) < 20:
                baseVals = detail[c,:]
            detailSigma[c] = _robustSigma(baseVals)

        zDetail = detail / detailSigma[:,None]
        groupDetailScore = np.sqrt(np.mean(zDetail**2,axis=0))
        # Expand guidance modestly at coarse scales, with a hard cap.
        basePad = int(round(coefficientPadSec * fs))
        nominalHalfSupport = int(round(0.5 * (filterLength - 1) * (2**(levelNumber - 1))))
        maxPad = int(round(maxWaveletPadSec * fs))
        levelPad = min(maxPad,basePad + nominalHalfSupport)
        guideLevel = _dilateMask1D(guidePad,levelPad)
        keepCoeff = guideLevel & (groupDetailScore >= waveletSigma)
        selectedCounts.append(int(np.sum(keepCoeff)))

        # Hard removal of selected detail coefficients in the CLEAN HP representation.
        for c in range(nComp):
            cAclean = cleanCoeffsByComp[c][coeffIndex][0]
            cDclean = cleanCoeffsByComp[c][coeffIndex][1]
            cDclean[keepCoeff] = 0.0
            cleanCoeffsByComp[c][coeffIndex] = (cAclean,cDclean)

    # 11. Inverse SWT: model = original HP - cleaned HP
    modelLocal = np.zeros((nLocal,nComp),dtype=np.float64)
    cleanedHPLocal = np.zeros((nLocal,nComp),dtype=np.float64)
    for c in range(nComp):
        cleanedHPPad = pywt.iswt(cleanCoeffsByComp[c],wavelet,norm=False)
        modelPad = YhpPad[:,c] - cleanedHPPad
        iCrop1 = padLeft
        iCrop2 = padLeft + nLocal
        modelLocal[:,c] = modelPad[iCrop1:iCrop2]
        cleanedHPLocal[:,c] = cleanedHPPad[iCrop1:iCrop2]

    # 12. Subtract the HP glitch model from the untouched raw input
    cleanedLocal = Yraw - modelLocal
    background = Yraw - Yhp

    # 13. HP variance reduction within detected transient regions
    evalMask = glitchGuideMask
    hpSSE0 = np.sum(Yhp[evalMask,:]**2,axis=0)
    hpSSE1 = np.sum((Yhp[evalMask,:] - modelLocal[evalMask,:])**2,axis=0)
    componentVR = 1.0 - hpSSE1 / np.maximum(hpSSE0,1.0e-20)
    groupVR = float(np.mean(componentVR))

    # 14. Raw diagnostic variance reduction after removing local polynomial
    tBackground = np.linspace(-1.0,1.0,nLocal)
    B = np.column_stack([tBackground**p for p in range(backgroundDegree + 1)])
    Bpinv = np.linalg.pinv(B,rcond=pinvRcond)
    rawProj = Yraw - B @ (Bpinv @ Yraw)
    cleanedProj = cleanedLocal - B @ (Bpinv @ cleanedLocal)
    rawSSE0 = np.sum(rawProj[evalMask,:]**2,axis=0)
    rawSSE1 = np.sum(cleanedProj[evalMask,:]**2,axis=0)
    rawComponentVR = 1.0 - rawSSE1 / np.maximum(rawSSE0,1.0e-20)
    rawGroupVR = float(np.mean(rawComponentVR))

    # 15. Compatibility amplitudes: HP samples at packet maxima, not physical steps
    amplitudes = np.zeros((nComp,nPackets),dtype=np.float64)
    for k,iPeak in enumerate(packetPeakLocal):
        amplitudes[:,k] = Yhp[iPeak,:]

    # 16. Return all original output keys unchanged
    return {
        "search_interval": (i1,i2),
        "fit_interval": (fit1,fit2),
        "n_selected": nPackets,  # Number of detected transient packets
        "selected_indices": selectedIndices,
        "selected_samples_float": selectedIndices.astype(np.float64),
        "amplitudes": amplitudes,
        "component_vr": componentVR,
        "local_group_vr": groupVR,
        "transient_component_vr": componentVR,
        "transient_group_vr": groupVR,
        "raw_component_vr": rawComponentVR,
        "raw_group_vr": rawGroupVR,
        "model_local": modelLocal,
        "cleaned_local": cleanedLocal,
        "background": background,
        "glitch_mask_local": glitchCoreMask,
        "glitch_guide_mask_local": glitchGuideMask,
        "transient_score": transientScore,
        "hp_sigma": hpSigma,
        "packet_intervals": packetIntervalsGlobal,
        "packet_peak_scores": packetPeakScores,
        "wavelet_selected_counts": np.asarray(selectedCounts,dtype=int),
        "n_wavelet_coefficients": int(np.sum(selectedCounts)),
        "wavelet": wavelet,
        "swt_level": swtLevel,
        "wavelet_sigma": waveletSigma,
        "high_sigma": highSigma,
        "low_sigma": lowSigma,
        # Old compatibility diagnostics; these have no SWT-specific physical meaning.
        "event_correlations": np.full(nPackets,np.nan),
        "incremental_vr_history": np.array([],dtype=np.float64)
    }


def cleanGlitchesG2SWT(streamAllStage1,componentIndices,searchIntervals,hourStart,extractLen=3600.0,
                       overlapBuffer=60.0,highpassFreq=0.5,filterOrder=4,fitPadSec=2.0,highSigma=5.0,lowSigma=2.5,
                       scoreSmoothSec=0.075,bridgeSec=0.075,maskPadSec=0.10,wavelet="sym4",swtLevel=5,
                       waveletSigma=3.0,coefficientPadSec=0.10,maxWaveletPadSec=0.50,backgroundDegree=3,
                       pinvRcond=1.0e-10):
    """
    Full Stage-2 HP-guided SWT cleaner for one station.

    Output structure deliberately mirrors the previous Stage-2
    implementations.
    """

    startTime = (hourStart- overlapBuffer)
    validIndices = []
    xAnalysis = []
    fsUse = None
    nSamplesUse = None

    # =========================================================
    # 1. Common analysis-grid traces
    # =========================================================

    for indNow in componentIndices:
        if indNow is None:
            continue
        if (indNow < 0 or indNow >= len(streamAllStage1)):
            continue
        stNow = (streamAllStage1[indNow])
        if (stNow is None or len(stNow) == 0):
            continue
        st = stNow.copy()
        st.merge(method=1,fill_value=0)
        if len(st) != 1:
            continue
        tr = st[0]
        fs = float(tr.stats.sampling_rate)
        tr.trim(starttime=startTime,endtime=hourStart + extractLen,nearest_sample=False,pad=False)
        nExpected = int(round((overlapBuffer+ extractLen)*fs))
        if len(tr.data) < nExpected:
            continue

        x = np.asarray(tr.data[:nExpected],dtype=np.float64)
        if fsUse is None:
            fsUse = fs
            nSamplesUse = len(x)
        else:
            if not np.isclose(fs,fsUse,rtol=1.0e-6):
                raise ValueError("Different component sampling rates.")
            if len(x) != nSamplesUse:
                raise ValueError("Different component analysis lengths.")
        validIndices.append(indNow)
        xAnalysis.append(x)
    if len(validIndices) == 0:

        return {
            "status":
                "no_data",
            "component_indices":
                [],
            "window_results":
                [],
            "models_by_stream":
                {},
            "cleaned_by_stream":
                {}
        }
    fs = fsUse

    xAnalysis = np.asarray(xAnalysis,dtype=np.float64)
    nComp, nAnalysis = (xAnalysis.shape)
    # =========================================================
    # 2. High-pass COMPLETE analysis data once
    # =========================================================

    sosHP = sp.butter(filterOrder,highpassFreq,btype="highpass",fs=fs,output="sos")
    xHPAll = sp.sosfiltfilt(sosHP,xAnalysis,axis=1)
    # =========================================================
    # 3. Full analysis-grid glitch model
    # =========================================================

    modelAnalysis = np.zeros_like(xAnalysis)
    windowResults = []
    allPacketIndices = []
    allPacketAmplitudes = [[]
        for _ in range(
            nComp
        )
    ]
    
    # Independent G2 peak times for each component
    peakIndicesByStream = {indNow: [] for indNow in validIndices}
    
    # =========================================================
    # 4. Process every existing G2 interval independently
    # =========================================================

    for windowNo, interval in enumerate(searchIntervals):
        result = (solveHPGuidedSWTWindow(xAll=xAnalysis,xHPAll=xHPAll,interval=interval,fs=fs,
                                         fitPadSec=fitPadSec,highSigma=highSigma,lowSigma=lowSigma,scoreSmoothSec=scoreSmoothSec,
                                         bridgeSec=bridgeSec,maskPadSec=maskPadSec,wavelet=wavelet,swtLevel=swtLevel,
                                         waveletSigma=waveletSigma,coefficientPadSec=coefficientPadSec,maxWaveletPadSec=
                                         maxWaveletPadSec,backgroundDegree=backgroundDegree,pinvRcond=pinvRcond))
        if result is None:
            continue
        # ============================================================
        # Independent G2 packet peaks for each component
        # ============================================================

        fit1, fit2 = result["fit_interval"]
        i1, i2 = result["search_interval"]

        samples = np.arange(fit1, fit2 + 1)

        searchMask = (samples >= i1) & (samples <= i2)
        backgroundMask = ~searchMask

        if np.sum(backgroundMask) < max(20, int(fs)):
            backgroundMask = np.ones(len(samples), dtype=bool)

        nSmooth = max(1, int(round(scoreSmoothSec * fs)))
        nBridge = max(0, int(round(bridgeSec * fs)))

        componentPeaks = {}

        for c, indNow in enumerate(validIndices):

            # High-pass acceleration for this component only
            y = xHPAll[c, fit1:fit2 + 1]

            baseline = y[backgroundMask]

            center = np.median(baseline)
            sigma = _robustSigma(baseline)

            peaks = []

            if np.isfinite(sigma) and sigma > 0:

                z = (y - center) / sigma

                scoreInstant = np.abs(z)
                scoreSmooth = _movingRMS(scoreInstant, nSmooth)

                transientScore = np.maximum(scoreInstant,scoreSmooth)
                highMask = ((transientScore >= highSigma)& searchMask)
                lowMask = ((transientScore >= lowSigma)& searchMask)
                lowMask = _fillShortGaps(lowMask, nBridge)
                lowMask &= searchMask

                for p1, p2 in _connectedRegions1D(lowMask):

                    if not np.any(highMask[p1:p2 + 1]):
                        continue

                    iPeak = p1 + int(np.argmax(transientScore[p1:p2 + 1]))
                    # Convert local index to common analysis-grid index
                    peaks.append(fit1 + iPeak)

            componentPeaks[indNow] = np.asarray(peaks, dtype=int)
            peakIndicesByStream[indNow].extend(peaks)

        # Keep component-specific results for this G2 window
        result["component_peak_indices"] = componentPeaks

        result["component_peak_times_hour"] = {ind: indices / fs - overlapBuffer
            for ind, indices in componentPeaks.items()
        }
        result["window_number"] = windowNo
        result["times_trace"] = (result["selected_samples_float"]/fs)
        result["times_hour"] = (result["times_trace"]- overlapBuffer)
        windowResults.append(result)
        fit1, fit2 = (result["fit_interval"])
        # -----------------------------------------------------
        # Add reconstructed glitch component to hour model.
        #
        # G2 intervals should normally be disjoint. Since the
        # SWT model is strongly localized by the HP mask,
        # modest fitting-window overlap is generally harmless.
        # -----------------------------------------------------

        modelAnalysis[:,fit1:fit2 + 1] += (result["model_local"].T)
        if result["n_selected"] > 0:
            allPacketIndices.extend(
                result["selected_indices"].tolist())
            for c in range(nComp):
                allPacketAmplitudes[c].extend(
                    result["amplitudes"][c].tolist())
    
    # ============================================================
    # Combine peak detections across all components and windows
    # ============================================================

    peakIndicesByStream = {ind: np.unique(np.asarray(indices, dtype=int))
        for ind, indices in peakIndicesByStream.items()
    }

    peakArrays = [indices
        for indices in peakIndicesByStream.values()
        if len(indices) > 0
    ]

    if peakArrays:
        mergedPeakIndices = np.unique(np.concatenate(peakArrays))
    else:
        mergedPeakIndices = np.array([], dtype=int)

    peakTimesHourByStream = {ind: indices / fs - overlapBuffer
        for ind, indices in peakIndicesByStream.items()
    }

    mergedPeakTimesHour = (mergedPeakIndices / fs - overlapBuffer)
    # =========================================================
    # 5. Sort packet diagnostics
    # =========================================================

    if len(allPacketIndices) > 0:
        allPacketIndices = np.asarray(allPacketIndices,dtype=int)
        order = np.argsort(allPacketIndices)
        allPacketIndices = (allPacketIndices[order])
        for c in range(nComp):
            allPacketAmplitudes[c] = (np.asarray(allPacketAmplitudes[c],dtype=np.float64)[order])
    else:
        allPacketIndices = np.array([],dtype=int)
        for c in range(nComp):
            allPacketAmplitudes[c] = (np.array([],dtype=np.float64))
    # =========================================================
    # 6. Place analysis-grid model onto FULL raw traces
    # =========================================================

    modelsByStream = {}

    cleanedByStream = {}

    indicesByStream = {}

    amplitudesByStream = {}

    for c, indNow in enumerate(validIndices):
        trFull = (streamAllStage1[indNow][0])
        rawFull = np.asarray(trFull.data,dtype=np.float64)
        fullModel = np.zeros(len(rawFull),dtype=np.float64)
        iFull0 = int(round(float(startTime- trFull.stats.starttime)*fs))
        src1 = 0
        dst1 = (iFull0)
        if dst1 < 0:
            src1 = (-dst1)
            dst1 = 0
        nCanWrite = min(nAnalysis- src1,len(fullModel)- dst1)
        if nCanWrite > 0:
            fullModel[dst1:dst1 + nCanWrite] = (modelAnalysis[c,src1:src1 + nCanWrite])
        cleaned = (rawFull- fullModel)
        modelsByStream[indNow] = (fullModel)
        cleanedByStream[indNow] = (cleaned)
        indicesByStream[indNow] = (allPacketIndices.copy())
        amplitudesByStream[indNow] = (allPacketAmplitudes[c].copy())
    # =========================================================
    # 7. Return same outer structure as previous Stage 2
    # =========================================================

    return {"status":"ok","component_indices":validIndices,"startTime":startTime,"fs":fs,
            "window_results":windowResults,"indices_by_stream":indicesByStream,"amplitudes_by_stream":
            amplitudesByStream,"models_by_stream":modelsByStream,"cleaned_by_stream":cleanedByStream,
            "model_analysis":modelAnalysis,
            # Independent component peak detections
            "peak_indices_by_stream": peakIndicesByStream,
            "peak_times_hour_by_stream": peakTimesHourByStream,
            # Chronological union of BH1, BH2 and BHZ peaks
            "merged_peak_indices": mergedPeakIndices,
            "merged_peak_times_hour": mergedPeakTimesHour}


def mergeG2ComponentPeakTimes(stage2Result, coincidenceSec=0.5,
                              timeRange=None):
    """Return representative shared onsets and provenance of their detections.

    Parameters
    ----------
    stage2Result : dict
        One station's stage2DispAll[dNo], containing the new field
        ``peak_times_hour_by_stream`` (dict: stream index -> times in seconds).
    coincidenceSec : float
        Maximum time span *within each group*, not simply the distance
        between adjacent sorted peaks.  Prevents chaining a long sequence.
    timeRange : tuple or None
        Optional inclusive range of peak times relative to hour start.

    Returns
    -------
    dict with keys times_hour, clusters, peak_times_hour_by_stream,
    n_input_peaks and coincidence_sec.
    """
    if not np.isfinite(coincidenceSec) or coincidenceSec < 0:
        raise ValueError('coincidenceSec must be a nonnegative finite value')
    if stage2Result is None or 'peak_times_hour_by_stream' not in stage2Result:
        raise ValueError('Stage-2 result needs peak_times_hour_by_stream. '
                         'Run the updated per-component SWT peak extraction.')
    if timeRange is not None:
        if len(timeRange) != 2 or timeRange[1] <= timeRange[0]:
            raise ValueError('timeRange must be (minTime, maxTime)')

    records = []
    by_stream = {}
    for stream_index, values in stage2Result['peak_times_hour_by_stream'].items():
        idx = int(stream_index)
        a = np.asarray(values, dtype=float).ravel()
        if a.size and not np.all(np.isfinite(a)):
            raise ValueError(f'Nonfinite G2 peaks in stream {idx}')
        a = np.unique(a)
        if timeRange is not None:
            a = a[(a >= timeRange[0]) & (a <= timeRange[1])]
        by_stream[idx] = a
        records.extend((float(x), idx) for x in a)

    records.sort(key=lambda p: (p[0], p[1]))
    groups = []
    group = []
    for time_sec, idx in records:
        # COMPLETE-LINK span: a chain of adjacent events cannot become a
        # much longer physical event merely because each gap is small.
        if group and (time_sec - group[0][0]) > coincidenceSec + 1e-12:
            groups.append(group)
            group = []
        group.append((time_sec, idx))
    if group:
        groups.append(group)

    clusters = []
    for k, group in enumerate(groups):
        ts = np.asarray([a for a, _ in group], dtype=float)
        sources = sorted(set(idx for _, idx in group))
        clusters.append(dict(
            event_index=k,
            time_hour=float(np.median(ts)),
            first_peak_hour=float(ts[0]),
            last_peak_hour=float(ts[-1]),
            span_sec=float(np.ptp(ts)),
            source_streams=sources,
            n_detections=len(group),
            original_peaks=[{'time_hour': float(t), 'stream_index': int(i)}
                            for t, i in group],
        ))

    return dict(
        times_hour=np.asarray([c['time_hour'] for c in clusters], dtype=float),
        clusters=clusters,
        peak_times_hour_by_stream=by_stream,
        n_input_peaks=len(records),
        coincidence_sec=float(coincidenceSec),
    )


def _response(t, onset, g, tg):
    return np.interp(t - onset, tg, g, left=0.0, right=0.0)


def _raw_model(full_t, fs, times, amplitudes, g, tg):
    """Build the complete unfiltered step response at native sampling."""
    model = np.zeros(len(full_t), dtype=float)
    for event_time, amp in zip(times, amplitudes):
        left, right = event_time+tg[0], event_time+tg[-1]
        j0 = max(0, int(np.floor((left-full_t[0])*fs))-1)
        j1 = min(len(full_t), int(np.ceil((right-full_t[0])*fs))+2)
        if j1 > j0 and amp:
            model[j0:j1] += amp*_response(full_t[j0:j1], event_time, g, tg)
    return model


def _squared(z):
    return float(np.sum(np.asarray(z, dtype=float)**2))


def _ridge_solve(A, y, ridgeFraction):
    if A.shape[1] == 0:
        return np.empty(0, dtype=float)
    scale = max(_squared(A) / A.shape[1], np.finfo(float).tiny)
    ridge = float(ridgeFraction) * scale
    if ridge:
        a = np.vstack([A, np.sqrt(ridge) * np.eye(A.shape[1])])
        b = np.r_[y, np.zeros(A.shape[1])]
    else:
        a, b = A, y
    sol, *_ = np.linalg.lstsq(a, b, rcond=1e-10)
    return sol


def _evaluate_baseline(Aaug, Baug, target_aug, coeff):
    """Unpenalized nuisance baseline, estimated conditional on coefficients."""
    return np.linalg.lstsq(Baug, target_aug - Aaug @ coeff, rcond=None)[0]


def fitSharedG2OnsetsOneComponent(
        streamAll, stage2Result, template, tTemplate, hourStart,
        componentIndex, overlapBuffer=60.0,
        timeRange=(1480., 1540.), fitMarginSec=35.,
        coincidenceSec=0.5, lowpass=0.35, fitFs=5.0,
        derivativeWeightSec=2.0, ridgeFraction=1e-3,
        minSequentialSegmentSec=1.0, maxEvents=1000,
        mergedPeaks=None, verbose=True):
    """Fit raw-count acceleration-step responses at common G2 candidate times.

    One component per invocation (e.g. BH1: componentIndex=0).  Onset times
    are SHARED across components; amplitudes are solved independently.

    timeRange restricts which G2 peaks are fitted; the waveform objective is
    evaluated over [timeRange[0]-fitMarginSec, timeRange[1]+fitMarginSec].
    For excursions with older contributing steps, EXPAND timeRange backwards;
    an event omitted from the candidates cannot be reconstructed correctly.

    No timing optimization is done; events stay at the merged median times.
    There is NO greedy deletion: all events have a fitted coefficient (possibly
    very close to zero).  This is G2-only, not G1-plus-residual fitting.

    Output ``model`` is the physical raw-count step reconstruction, while
    ``baseline_fit`` is only an offset/trend nuisance term, defined on t_fit.
    Only ``model`` is subtracted from raw data in ``cleaned_streams``.
    """
    if stage2Result is None or stage2Result.get('status') != 'ok':
        raise ValueError('stage2Result must be a valid single-station G2 result')
    idx = int(componentIndex)
    if idx < 0 or idx >= len(streamAll) or streamAll[idx] is None or len(streamAll[idx]) != 1:
        raise ValueError(f'componentIndex={idx} has no single raw trace')
    tr = streamAll[idx][0]
    fs = float(tr.stats.sampling_rate)
    if not np.isclose(fs, float(stage2Result['fs']), rtol=1e-5):
        raise ValueError('Raw trace sample rate differs from Stage-2 sample rate')
    if not (0 < lowpass < min(fs, fitFs)/2):
        raise ValueError('lowpass must be below both Nyquist frequencies')
    if derivativeWeightSec < 0 or ridgeFraction < 0 or fitMarginSec < 0:
        raise ValueError('Invalid derivative, ridge, or fit-margin parameter')
    if fitFs <= 0 or minSequentialSegmentSec <= 0 or maxEvents <= 0:
        raise ValueError('fitFs, minSequentialSegmentSec, maxEvents must be positive')
    if timeRange is None or len(timeRange) != 2 or timeRange[1] <= timeRange[0]:
        raise ValueError('Supply a finite (start, end) timeRange for this local test')

    if mergedPeaks is None:
        mergedPeaks = mergeG2ComponentPeakTimes(
            stage2Result, coincidenceSec=coincidenceSec, timeRange=timeRange)
    else:
        if 'times_hour' not in mergedPeaks:
            raise ValueError('mergedPeaks must be from mergeG2ComponentPeakTimes')
    times = np.asarray(mergedPeaks['times_hour'], dtype=float)
    # Allow a precomputed union from a wider time span, but preserve this
    # experiment's explicit timeRange and sort the shared onsets.
    times = np.unique(times[(times >= timeRange[0]) & (times <= timeRange[1])])
    if len(times) > maxEvents:
        raise ValueError(f'{len(times)} merged events exceed maxEvents={maxEvents}')
    if len(times) == 0:
        raise ValueError('No independent G2 peaks in selected timeRange')

    g = np.asarray(template, dtype=np.float64).copy()
    tg = np.asarray(tTemplate, dtype=np.float64)
    if g.ndim != 1 or g.size < 6 or tg.shape != g.shape:
        raise ValueError('template and tTemplate must be matching 1D arrays')
    if np.any(~np.isfinite(g)) or np.any(~np.isfinite(tg)) or np.any(np.diff(tg) <= 0):
        raise ValueError('Template values must be finite and tTemplate strictly increasing')
    if not np.isclose(1.0/np.median(np.diff(tg)), fs, rtol=1e-3):
        raise ValueError('Template sample rate does not match the raw data')
    # Match convention of the previous G2-guided reconstructions.
    before = g[tg < 0]
    g -= float(np.median(before)) if before.size else float(g[0])

    raw = np.asarray(tr.data, dtype=float)
    if not np.all(np.isfinite(raw)):
        raise ValueError('Nonfinite raw counts: process gaps before this fit')
    full_t = float(tr.stats.starttime-hourStart) + np.arange(len(raw))/fs
    lo = max(float(full_t[0]), float(timeRange[0])-fitMarginSec)
    hi = min(float(full_t[-1]), float(timeRange[1])+fitMarginSec)
    if hi <= lo:
        raise ValueError('No fit interval within raw trace coverage')
    if times.min() < lo or times.max() > hi:
        raise ValueError('One or more merged onsets are outside the fit interval')
    tfit = np.arange(lo, hi+0.25/fitFs, 1.0/fitFs)
    if len(tfit) < 10:
        raise ValueError('Fitting interval is too short')
    if tfit[-1] > full_t[-1]:
        tfit = tfit[tfit <= full_t[-1]]

    sos = signal.butter(4, lowpass, btype='lowpass', fs=fs, output='sos')
    target = np.interp(tfit, full_t, signal.sosfiltfilt(sos, raw))
    # Work with the lowpass template just as in the prior derivative model.
    gf = signal.sosfiltfilt(sos, g)
    A = np.column_stack([_response(tfit, onset, gf, tg) for onset in times])

    tcenter = float(0.5*(tfit[0] + tfit[-1]))
    thalf = max(float(0.5*(tfit[-1]-tfit[0])), 1.0)
    B = np.column_stack([np.ones(len(tfit)), (tfit-tcenter)/thalf])
    beta = float(derivativeWeightSec)
    if beta:
        d_target = np.gradient(target, tfit, edge_order=2)
        d_A = np.gradient(A, tfit, axis=0, edge_order=2)
        d_B = np.gradient(B, tfit, axis=0, edge_order=2)
        Aaug = np.vstack((A, beta*d_A))
        Baug = np.vstack((B, beta*d_B))
        yaug = np.r_[target, beta*d_target]
    else:
        d_target = np.gradient(target, tfit, edge_order=2)
        Aaug, Baug, yaug = A, B, target

    # Estimate initial nuisance baseline from the pre-event segment only,
    # if a sufficient pre-onset window exists.
    before_mask = tfit < times[0] - 0.5
    good_baseline = bool(np.count_nonzero(before_mask) >= max(8, int(2.0*fitFs)))
    if good_baseline:
        baseline_initial, *_ = np.linalg.lstsq(
            B[before_mask], target[before_mask], rcond=None)
    else:
        # Weak initialization, NOT a license to remove this baseline.
        baseline_initial, *_ = np.linalg.lstsq(B, target, rcond=None)

    # ---------------------------------------------------------------
    # Sequential peak-to-peak estimate.  This is NOT the final fit:
    # each new step is solved only from [its onset, next onset), with
    # all previous step responses continuing to evolve.
    # ---------------------------------------------------------------
    initial_amplitudes = np.zeros(len(times), dtype=float)
    sequential_notes = []
    residual_init = yaug - Baug @ baseline_initial
    nfit = len(tfit)
    for k, onset in enumerate(times):
        next_onset = times[k+1] if k+1 < len(times) else tfit[-1]+1e-6
        interval = (tfit >= onset) & (tfit < next_onset)
        ns = int(np.count_nonzero(interval))
        if ns < max(3, int(np.ceil(minSequentialSegmentSec*fitFs))):
            sequential_notes.append(dict(event_index=k, n_fit_samples=ns,
                                         status='too_short', amplitude=0.0))
            continue
        # Use the same samples in the derivative half of the objective.
        mask_aug = np.r_[interval, interval] if beta else interval
        col = Aaug[mask_aug, k]
        r = residual_init[mask_aug]
        denom = float(col @ col)
        if denom < 1e-18 or not np.isfinite(denom):
            sequential_notes.append(dict(event_index=k, n_fit_samples=ns,
                                         status='no_local_response', amplitude=0.0))
            continue
        amp = float((col @ r)/denom)
        initial_amplitudes[k] = amp
        residual_init -= Aaug[:, k]*amp
        sequential_notes.append(dict(event_index=k, n_fit_samples=ns,
                                     status='estimated', amplitude=amp))

    # ---------------------------------------------------------------
    # Joint ridge solution: all event amplitudes can readjust, while
    # staying at the SAME onset times for this channel and every other.
    # Offset/slope are *nuisances* and remain outside the step model.
    # ---------------------------------------------------------------
    nuisance_pinv = np.linalg.pinv(Baug)
    y_proj = yaug - Baug @ (nuisance_pinv @ yaug)
    A_proj = Aaug - Baug @ (nuisance_pinv @ Aaug)
    amplitudes = _ridge_solve(A_proj, y_proj, ridgeFraction)
    bfit = _evaluate_baseline(Aaug, Baug, yaug, amplitudes)
    b0 = nuisance_pinv @ yaug  # best baseline-only comparison

    # Full native sample rate PHYSICAL acceleration-step corrections.
    sequential_model = _raw_model(full_t, fs, times, initial_amplitudes, g, tg)
    model = _raw_model(full_t, fs, times, amplitudes, g, tg)
    cleaned = raw - model
    # Score the ACTUAL correction generated on the original sample grid.
    actual_step_lp = np.interp(tfit, full_t, signal.sosfiltfilt(sos, model))
    residual_only_baseline = target - B @ b0
    residual_joint = target - actual_step_lp - B @ bfit
    d0 = np.gradient(residual_only_baseline, tfit, edge_order=2)
    d1 = np.gradient(residual_joint, tfit, edge_order=2)
    J0 = _squared(residual_only_baseline) + beta*beta*_squared(d0)
    J1 = _squared(residual_joint) + beta*beta*_squared(d1)
    local = (tfit >= timeRange[0]) & (tfit <= timeRange[1])

    result_streams = [None if st is None else st.copy() for st in streamAll]
    result_streams[idx][0].data = cleaned.astype(np.float64)

    diagnostics = dict(
        component_index=idx,
        n_original_peaks=mergedPeaks['n_input_peaks'],
        n_merged_onsets=len(times),
        n_nonzero_amplitudes=int(np.count_nonzero(np.abs(amplitudes) >
            max(np.max(np.abs(amplitudes))*1e-4, 1e-30))),
        fit_range=(float(tfit[0]), float(tfit[-1])),
        coefficient_l2_norm=float(np.linalg.norm(amplitudes)),
        low_frequency_gain_vs_baseline_only=1-_squared(residual_joint)/max(_squared(residual_only_baseline),1e-30),
        derivative_gain_vs_baseline_only=1-_squared(d1)/max(_squared(d0),1e-30),
        weighted_gain_vs_baseline_only=1-J1/max(J0,1e-30),
        local_low_frequency_gain_vs_baseline_only=1-_squared(residual_joint[local])/max(_squared(residual_only_baseline[local]),1e-30),
        local_derivative_gain_vs_baseline_only=1-_squared(d1[local])/max(_squared(d0[local]),1e-30),
        initial_baseline_has_pre_event_data=good_baseline,
        baseline_only_coefficients=b0.copy(),
        fitted_baseline_coefficients=bfit.copy(),
        step_and_baseline_fit_different_from_step_only=True,
    )
    if verbose:
        print(f'Stream {idx}: {mergedPeaks["n_input_peaks"]} individual G2 peaks '
              f'-> {len(times)} shared onsets (within {coincidenceSec:.2f} s)')
        print(f'  Weighted LF+slope gain over baseline-only: '
              f'{diagnostics["weighted_gain_vs_baseline_only"]:.4f}')
        print(f'  LF gain: {diagnostics["low_frequency_gain_vs_baseline_only"]:.4f}, '
              f'derivative gain: {diagnostics["derivative_gain_vs_baseline_only"]:.4f}')
        if not good_baseline:
            print('  WARNING: Insufficient quiet data preceding the first event; '
                  'baseline and early steps may be degenerate.')
    return dict(
        component_index=idx,
        onset_times_hour=times.copy(),
        amplitudes=amplitudes.copy(),
        sequential_amplitudes=initial_amplitudes,
        sequential_model=sequential_model,
        sequential_notes=sequential_notes,
        merged_peaks=mergedPeaks,
        models_by_stream={idx: model},
        cleaned_streams=result_streams,
        model=model,
        cleaned_counts=cleaned,
        fit_time=tfit,
        raw_lowpass=target,
        model_lowpass=actual_step_lp,
        baseline_fit=B @ bfit,
        baseline_only=B @ b0,
        fitted_lowpass_signal=actual_step_lp + B @ bfit,
        diagnostics=diagnostics,
    )