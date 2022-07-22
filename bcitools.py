#!/usr/bin/env python
# coding: utf-8

# 1. Data load
#     1. load
#     2. s to cnt
#     3. label index (1\~4 -> 0\~3)
#     4. pos index (start from 0)

# In[1]:


from scipy.io import loadmat
import mne
import numpy as np
from copy import deepcopy
from functools import wraps
from collections import OrderedDict


def verbose_func_name(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        if ("verbose" in kwargs.keys()) and kwargs["verbose"]:
            print("\n" + fn.__name__)
        return fn(*args, **kwargs)

    return inner


# @verbose_func_name
# def load_gdf2mat(subject, train=True, data_dir=".", overflowdetection=True, verbose=False):
#     # Configuration
#     if train:
#         filename = f"A0{subject}T_gdf"
#     else:
#         filename = f"A0{subject}E_gdf"
#     base = data_dir

#     # Load mat files
#     data_path  =  base + '/gdf2mat/' + filename + '.mat'
#     label_path =  base + '/true_labels/' + filename[:4] + '.mat'

#     if not overflowdetection:
#         filename  = filename + "_overflowdetection_off"
#         data_path = base + '/gdf2mat_overflowdetection_off/' + filename + '.mat'

#     session_data = loadmat(data_path, squeeze_me=False)
#     label_data   = loadmat(label_path, squeeze_me=False)

#     # Parse data
#     s = session_data["s"] # signal
#     h = session_data["h"] # header
#     labels = label_data["classlabel"] # true label

#     h_names = h[0][0].dtype.names # header is structured array
#     origin_filename = h["FileName"][0,0][0]
#     train_labels = h["Classlabel"][0][0] # For Evaluation data, it is filled with NaN.
#     artifacts = h["ArtifactSelection"][0,0]

#     events = h['EVENT'][0,0][0,0] # void
#     typ = events['TYP']
#     pos = events['POS']
#     fs  = events['SampleRate'].squeeze()
#     dur = events['DUR']

#     # http://www.bbci.de/competition/iv/desc_2a.pdf
#     typ2desc = {276:'Idling EEG (eyes open)',
#                 277:'Idling EEG (eyes closed)',
#                 768:'Start of a trial',
#                 769:'Cue onset left (class 1)',
#                 770:'Cue onset right (class 2)',
#                 771:'Cue onset foot (class 3)',
#                 772:'Cue onset tongue (class 4)',
#                 783:'Cue unknown',
#                 1024:'Eye movements',
#                 32766:'Start of a new run'}

#     # 출처... 아마... brain decode...
#     ch_names = ['Fz',  'FC3', 'FC1', 'FCz', 'FC2',
#                  'FC4', 'C5',  'C3',  'C1',  'Cz',
#                  'C2',  'C4',  'C6',  'CP3', 'CP1',
#                  'CPz', 'CP2', 'CP4', 'P1',  'Pz',
#                  'P2',  'POz', 'EOG-left', 'EOG-central', 'EOG-right']

#     assert filename[:4] == origin_filename[:4]
#     if verbose:
#         print("- filename:", filename)
#         print("- load data from:", data_path)
#         print('\t- original fileanme:', origin_filename)
#         print("- load label from:", label_path)
#         print("- shape of s", s.shape) # (time, 25 channels),
#         print("- shape of labels", labels.shape) # (288 trials)

#     data =  {"s":s, "h":h, "labels":labels, "filename":filename, "artifacts":artifacts, "typ":typ, "pos":pos, "fs":fs, "dur":dur, "typ2desc":typ2desc, "ch_names":ch_names}
#     return data


@verbose_func_name
def load_gdf2mat_feat_mne(
    subject, train=True, data_dir=".", overflowdetection=True, verbose=False
):
    # Configuration
    if train:
        filename = f"A0{subject}T_gdf"
    else:
        filename = f"A0{subject}E_gdf"
    base = data_dir

    assert (
        not overflowdetection
    ), "load_gdf2mat_feat_mne does not support overflowdetection..."

    # Load mat files
    data_path = (
        base
        + "/gdf2mat_overflowdetection_off/"
        + filename
        + "_overflowdetection_off.mat"
    )
    label_path = base + "/true_labels/" + filename[:4] + ".mat"

    session_data = loadmat(data_path, squeeze_me=False)
    label_data = loadmat(label_path, squeeze_me=False)

    gdf_data_path = base + "/" + filename[:4] + ".gdf"
    raw_gdf = mne.io.read_raw_gdf(gdf_data_path, stim_channel="auto")
    raw_gdf.load_data()

    # Parse data
    s = raw_gdf.get_data().T  # cnt -> tnc
    assert np.allclose(
        s * 1e6, session_data["s"]
    ), "mne and loadmat loaded different singal..."
    h = session_data["h"]  # header
    labels = label_data["classlabel"]  # true label

    h_names = h[0][0].dtype.names  # header is structured array
    origin_filename = h["FileName"][0, 0][0]
    train_labels = h["Classlabel"][0][0]  # For Evaluation data, it is filled with NaN.
    artifacts = h["ArtifactSelection"][0, 0]

    events = h["EVENT"][0, 0][0, 0]  # void
    typ = events["TYP"]
    pos = events["POS"]
    fs = events["SampleRate"].squeeze()
    dur = events["DUR"]

    # http://www.bbci.de/competition/iv/desc_2a.pdf
    typ2desc = {
        276: "Idling EEG (eyes open)",
        277: "Idling EEG (eyes closed)",
        768: "Start of a trial",
        769: "Cue onset left (class 1)",
        770: "Cue onset right (class 2)",
        771: "Cue onset foot (class 3)",
        772: "Cue onset tongue (class 4)",
        783: "Cue unknown",
        1024: "Eye movements",
        32766: "Start of a new run",
    }

    # 출처... 아마... brain decode...
    ch_names = [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
        "EOG-left",
        "EOG-central",
        "EOG-right",
    ]

    assert filename[:4] == origin_filename[:4]
    if verbose:
        print("- filename:", filename)
        print("- load data from:", data_path)
        print("\t- original fileanme:", origin_filename)
        print("- load label from:", label_path)
        print("- shape of s", s.shape)  # (time, 25 channels),
        print("- shape of labels", labels.shape)  # (288 trials)

    data = {
        "s": s,
        "h": h,
        "labels": labels,
        "filename": filename,
        "artifacts": artifacts,
        "typ": typ,
        "pos": pos,
        "fs": fs,
        "dur": dur,
        "typ2desc": typ2desc,
        "ch_names": ch_names,
    }
    return data


@verbose_func_name
def s_to_cnt(data, verbose=False):
    data = deepcopy(data)
    assert ("s" in data.keys()) and ("cnt" not in data.keys())
    data["cnt"] = data.pop("s").T

    if verbose:
        print("- shape of cnt:", data["cnt"].shape)
    return data


@verbose_func_name
def rerange_label_from_0(data, verbose=False):
    data = deepcopy(data)
    data["labels"] = data["labels"] - 1
    assert np.array_equal(np.unique(data["labels"]), [0, 1, 2, 3])

    if verbose:
        print("- unique labels:", np.unique(data["labels"]))
    return data


@verbose_func_name
def rerange_pos_from_0(data, verbose=False):
    """
    In matlab, index starts from 1.
    In python, index starts from 0.
    To adapt index type data, subtract 1 from it.
    """
    data = deepcopy(data)
    data["pos"] = data["pos"] - 1
    assert data["pos"].min() == 0

    if verbose:
        print("- initial value:", data["pos"][0])
        print("- minimum value:", np.min(data["pos"]))
    return data


# 2. Preprocessing
#     1. drop EOG channels
#     2. replace break with mean
#     3. scaling (microvolt)
#     4. bandpass 4-38Hz (butterworth 3rd order)
#     5. exponential running standardization (init_block_size=1000, factor_new=1e-3)
#     6. epoch (cue-0.5ms ~ cue+4ms)
#     * no rejection

# In[2]:


import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
from braindecode.datautil import (
    exponential_moving_standardize,  # moving은 최신, running은 예전 꺼. axis가 달라서 중요함!
)


@verbose_func_name
def drop_eog_from_cnt(data, verbose=False):
    assert (data["cnt"].shape[0] == 25) and (
        len(data["ch_names"]) == 25
    ), "the number of channels is not 25..."
    data = deepcopy(data)
    data["cnt"] = data["cnt"][0:22]
    data["ch_names"] = data["ch_names"][0:22]

    if verbose:
        print("- shape of cnt:", data["cnt"].shape)
    return data


@verbose_func_name
def replace_break_with_mean(data, verbose=False):
    data = deepcopy(data)
    cnt = data["cnt"]
    for i_chan in range(cnt.shape[0]):
        this_chan = cnt[i_chan]
        cnt[i_chan] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(cnt[i_chan])
        chan_mean = np.nanmean(cnt[i_chan])
        cnt[i_chan, mask] = chan_mean
    data["cnt"] = cnt
    assert not np.any(np.isnan(cnt)), "nan remains in cnt.."

    if verbose:
        print("- min of cnt:", np.min(cnt))
    return data


@verbose_func_name
def change_scale(data, factor, channels="all", verbose=False):
    """
    Args
    ----
    data : dict
    factor : float
    channels : list of int, or int
    verbose : bool
    """
    data = deepcopy(data)
    if channels == "all":
        channels = list(range(data["cnt"].shape[0]))
    elif isinstance(channels, int):
        channels = [channels]

    assert hasattr(channels, "__len__"), "channels should be list or int..."

    assert (max(channels) <= data["cnt"].shape[0]) and (
        min(channels) >= 0
    ), "channel index should be between 0 and #channel of data..."

    assert ("s" not in data.keys()) and ("cnt" in data.keys())

    data["cnt"][channels, :] = data["cnt"][channels, :] * factor

    if verbose:
        print("- applied channels:", channels)
        print("- factor :", factor)
        print("- maximum value:", np.max(data["cnt"]))
        print("- minimum value:", np.min(data["cnt"]))
    return data


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_lowpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="lowpass")
    return b, a


def butter_highpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype="highpass")
    return b, a


@verbose_func_name
def butter_bandpass_filter(data, lowcut=0, highcut=0, order=3, axis=-1, verbose=False):
    assert (lowcut != 0) or (
        highcut != 0
    ), "one of lowcut and highcut should be not 0..."
    data = deepcopy(data)
    fs = data["fs"]

    if lowcut == 0:
        print("banpass changes into lowpass " "since lowcut is 0 ...")
        b, a = butter_lowpass(highcut, fs, order)
    elif highcut == 0:
        print("bandpass changes into highpass " "since highcut is 0 ...")
        b, a = butter_highpass(lowcut, fs, order)
    else:
        b, a = butter_bandpass(lowcut, highcut, fs, order)

    data["cnt"] = lfilter(b, a, data["cnt"], axis=axis)
    if verbose:
        if lowcut == 0:
            print(f"- lowpass : {highcut}Hz")
        elif highcut == 0:
            print(f"- highpass : {lowcut}Hz")
        else:
            print(f"- {lowcut}-{highcut}Hz")
        print(f"- order {order}")
        print(f"- fs {fs}Hz")
    return data


@verbose_func_name
def exponential_moving_standardize_from_braindecode(
    data, factor_new, init_block_size, eps=1e-4, verbose=False
):
    """
    for latest braindecode version...
    exponential_moving_standardize takes cnt (time, channel)
    """
    data = deepcopy(data)
    before_mean = np.mean(data["cnt"], axis=1)
    data["cnt"] = exponential_moving_standardize(
        data["cnt"], factor_new=factor_new, init_block_size=init_block_size, eps=eps
    )
    assert np.all(before_mean != np.mean(data["cnt"], axis=1))
    if verbose:
        print("- factor_new", factor_new)
        print("- init_block_size", init_block_size)
        print("- mean before standarization")
        print(before_mean)
        print("- mean after  standarization")
        print(np.mean(data["cnt"], axis=1))
    return data


# @verbose_func_name
# def exponential_running_standardize_from_braindecode(data,
#                          factor_new,
#                          init_block_size,
#                          eps=1e-4,
#                          verbose=False):
#     """
#     for outdated braindecode version...
#     exponential_running_standardize takes tnc (time, channel)
#     """
#     data = deepcopy(data)
#     before_mean = np.mean(data["cnt"], axis=1)
#     data["cnt"] = exponential_running_standardize(
#         data["cnt"].T,
#         factor_new=factor_new,
#         init_block_size=init_block_size,
#         eps=eps
#     ).T
#     assert np.all(before_mean != np.mean(data["cnt"], axis=1))
#     if verbose:
#         print("- factor_new", factor_new)
#         print("- init_block_size", init_block_size)
#         print("- mean before standarization")
#         print(before_mean)
#         print("- mean after  standarization")
#         print(np.mean(data["cnt"], axis=1))
#     return data


@verbose_func_name
def epoch_X_y_from_data(data, start_sec_offset, stop_sec_offset, verbose=False):
    """
    Args
    ----
    data : dict
        It can be obtained by load_gdf2mat and s_to_cnt functions.
    start_sec_offset : int
    stop_sec_offset : int
    verbose : bool

    Return
    ------
    X : 3d array (n_trials, n_channels, time)
    y : 2d array (n_trials, 1)

    NOTE
    ----
    The base of offset is 'start of a trial onset'.
    NOT based on 'cue onset'. if you want to use offset
    based on 'cue onset', add 2 sec to start_sec_offset
    and stop_sec_offset.
    """
    cnt = data["cnt"]
    pos = data["pos"]
    typ = data["typ"]
    fs = data["fs"]

    start_onset = pos[typ == 768]  # start of a trial
    trials = []
    for i, onset in enumerate(start_onset):
        trials.append(
            cnt[
                0:22,
                int(onset + start_sec_offset * fs) : int(onset + stop_sec_offset * fs),
            ]  # start of a trial + 1.5 ~ 6
        )
    X = np.array(trials)  # trials, channels, time
    y = data["labels"]

    if verbose:
        print("- From : start of a trial onset +", start_sec_offset, "sec")
        print("- To   : start of a trial onset +", stop_sec_offset, "sec")
        print("- shape of X", X.shape)
        print("- shape of y", y.shape)

    return X, y


# 3. Split
#     - split train into train and validation (8:2)

# In[3]:


@verbose_func_name
def split_train_val(X, y, val_ratio, verbose=False):
    assert (val_ratio < 1) and (val_ratio > 0), "val_raion not in (0, 1)"
    val_size = round(len(y) * val_ratio)
    X_tr, y_tr = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]
    assert (len(X_tr) == len(y_tr)) and (
        len(X_val) == len(y_val)
    ), "each pair of X and y should have same number of trials..."
    assert len(X) == len(X_tr) + len(
        X_val
    ), "sum of number of splited trials should equal number of unsplited trials"
    if verbose:
        print("- shape of X_tr", X_tr.shape)
        print("- shape of y_tr", y_tr.shape)
        print("- shape of X_val", X_val.shape)
        print("- shape of y_val", y_val.shape)
    return X_tr, y_tr, X_val, y_val


# 4. Crop (bunch of crops)
#     - input_time_length: 1000 samples
#     * augmentation effect (twice)

# In[4]:


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')
