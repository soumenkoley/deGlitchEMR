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

# function to load all streams
def loadStreams(archiveAll, invAll, stnList, stnNames,locList, compList, netList, sTime, eTime,
                maxAttempts=3, retryWait=3.0):

    streamList = []
    stNo = 0

    for i in range(len(stnNames)):
        print('Doing stn = ' + str(stnNames[i]))

        for j in range(3):

            seedID = (f"{netList[stNo]}.{stnList[stNo]}."
                      f"{locList[stNo]}.{compList[stNo]}")

            # Default: empty stream if download fails
            streamNow = Stream()

            for attempt in range(1, maxAttempts + 1):

                try:
                    streamNow = archiveAll[i].get_waveforms(station=stnList[stNo],channel=compList[stNo],
                                                            network=netList[stNo],location=locList[stNo],
                                                            starttime=sTime,endtime=eTime)

                    if streamNow is None or len(streamNow) == 0:
                        print(f"No data: {seedID}")
                        streamNow = Stream()

                    # Successful request
                    break

                except FDSNNoDataException:
                    print(f"No data available: {seedID}")
                    streamNow = Stream()
                    break

                except Exception as error:
                    print(
                        f"Download failed: {seedID}, "
                        f"attempt {attempt}/{maxAttempts}"
                    )
                    print(f"  Time: {sTime} -> {eTime}")
                    print(f"  Error: {error}")

                    if attempt < maxAttempts:
                        time.sleep(retryWait)
                    else:
                        print(f"Skipping {seedID} for this interval.")
                        streamNow = Stream()

            # Always append, even when download fails
            streamList.append(streamNow)
            stNo += 1

    return streamList


def processStream(streamAll, hourStart, extractLen=3600):
    """
    Initially, only determine missing data for every stream.
    """

    missingAll = []

    for streamNow in streamAll:
        missingSeconds = getMissingSeconds(streamNow=streamNow,hourStart=hourStart,extractLen=extractLen)

        missingAll.append(missingSeconds)

    return missingAll

def prepareStream(streamAll,invAll,hourStart,extractLen=3600,dataBuffB=60,output="VEL",preFilt=None):
    """
    Remove the instrument response from all available streams.

    Each processed Trace remains at its native sampling rate. The returned
    Stream contains data from `dataBuffB` seconds before the analysis hour
    through the end of the analysis hour.

    Parameters
    ----------
    streamAll : list
        List containing ObsPy Streams or None.
    invAll : list
        Inventories corresponding to entries in streamAll. This assumes
        invAll has the same indexing as streamAll.
    hourStart : obspy.UTCDateTime
        Start of the actual one-hour analysis interval.
    extractLen : float
        Length of the analysis interval in seconds.
    dataBuffB : float
        Leading overlap buffer retained for PSD calculation.
    output : str
        Response-removal output, normally "VEL", "ACC", or "DISP".
    preFilt : tuple or None
        Four-corner frequency pre-filter for remove_response().

    Returns
    -------
    obspy.Stream
        One Stream containing all successfully processed traces.
    """

    processedStream = []

    returnStart = hourStart - dataBuffB
    returnEnd = hourStart + extractLen

    for streamNo, streamNow in enumerate(streamAll):

        # Preserve the missing channel as an absent trace.
        if streamNow is None or len(streamNow) == 0:
            processedStream.append(Stream())
            continue

        st = streamNow.copy()

        # Convert multiple segments to one continuous trace.
        # Gaps are zero-filled only for response removal.
        st.merge(method=1,fill_value=0,)
        inventoryNo = streamNo // 3

        try:
            st.remove_response(inventory=invAll[inventoryNo],output=output,pre_filt=preFilt,water_level=60,
                               taper=True,taper_fraction=0.05,zero_mean=True,hide_sensitivity_mismatch_warning=True)

        except Exception as error:
            print(f"Response removal failed for streamAll[{streamNo}]: "f"{error}")
            processedStream.append(Stream())
            continue

        # Keep the 60 s leading PSD buffer, but discard the longer
        # response-removal taper buffer.
        st.trim(starttime=returnStart,endtime=returnEnd,nearest_sample=False,pad=False)

        # Ensure ordinary NumPy arrays are returned.
        for tr in st:
            tr.data = np.asarray(tr.data, dtype=np.float64)

        processedStream.append(st)

    return processedStream