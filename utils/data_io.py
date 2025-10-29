from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Dict, Any, Tuple, List

def _mat_to_list(obj):
    """Convert MATLAB cell/struct to nested Python lists/dicts as needed."""
    # Attempt to handle common MATLAB cell formats saved via scipy.io.savemat
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [ _mat_to_list(x) for x in obj.squeeze().tolist() ]
    return obj

def load_data(data_file: str, channel_list=None, exclude_chans=None):
    """Load MATLAB .mat structured EEG data into a Python-friendly map.
    
    Expected variables in MAT:
      - EEG: 1xC cell, each cell = [group1, group2?] (each group is list of subject vectors)
      - Channel_location: 1xC list of channel names
      - Filenames (optional): struct with .group1 / .group2 containing IDs
    """
    S = loadmat(data_file, squeeze_me=True, struct_as_record=False)
    if 'EEG' not in S or 'Channel_location' not in S:
        raise ValueError(f"Expected variables EEG and Channel_location in {data_file}")
    EEG = _mat_to_list(S['EEG'])
    Channel_location = _mat_to_list(S['Channel_location'])
    if isinstance(Channel_location, np.ndarray):
        Channel_location = Channel_location.tolist()
    Channel_location = [str(x) for x in Channel_location]

    if channel_list is None or len(channel_list)==0:
        channel_list = Channel_location
    if exclude_chans is None:
        exclude_chans = []

    chan_set = {c.upper() for c in channel_list}
    excl_set = {c.upper() for c in exclude_chans}

    DataMap: Dict[str, Dict[str, list]] = {}
    SubjectIDs: Dict[str, Dict[str, np.ndarray]] = {}
    GroupNames = np.array(['group1','group2'], dtype=object)

    # Optional filenames struct
    Filenames = S.get('Filenames', None)

    for i, ch in enumerate(Channel_location):
        if ch.upper() not in chan_set or ch.upper() in excl_set:
            continue
        entry = {'group1': [], 'group2': []}
        cell_ch = EEG[i]
        # MATLAB convention: each channel cell contains 1 or 2 groups (lists of subjects)
        if isinstance(cell_ch, (list, tuple)) and len(cell_ch) >= 1:
            entry['group1'] = list(cell_ch[0]) if isinstance(cell_ch[0], (list, tuple, np.ndarray)) else [cell_ch[0]]
        if isinstance(cell_ch, (list, tuple)) and len(cell_ch) >= 2:
            entry['group2'] = list(cell_ch[1]) if isinstance(cell_ch[1], (list, tuple, np.ndarray)) else [cell_ch[1]]
        DataMap[ch] = entry

        # Subject IDs (fallback synthetic)
        sid_g1, sid_g2 = None, None
        if Filenames is not None and hasattr(Filenames, 'group1'):
            sid_g1 = np.array(Filenames.group1, dtype=object)
        if Filenames is not None and hasattr(Filenames, 'group2'):
            sid_g2 = np.array(Filenames.group2, dtype=object)
        if sid_g1 is None:
            sid_g1 = np.array([f"g1_{k+1}" for k in range(len(entry['group1']))], dtype=object)
        if sid_g2 is None:
            sid_g2 = np.array([f"g2_{k+1}" for k in range(len(entry['group2']))], dtype=object)
        SubjectIDs[ch] = {'group1': sid_g1.astype(str), 'group2': sid_g2.astype(str)}

    return DataMap, Channel_location, SubjectIDs, GroupNames

def read_labels_table(filepath: str) -> pd.DataFrame:
    """Read labels table with columns ID and Target (auto-detect common alternatives)."""
    if not filepath:
        return pd.DataFrame(columns=['ID','Target'])
    if filepath.lower().endswith(('.xlsx','.xls')):
        T = pd.read_excel(filepath)
    else:
        T = pd.read_csv(filepath)
    cols = {c.lower(): c for c in T.columns}
    if 'id' not in cols:
        raise ValueError('Clinical label file must contain an "ID" column.')
    if 'target' not in cols:
        for alt in ['days','moca','updrs','score','value']:
            if alt in cols:
                T['Target'] = T[cols[alt]]
                break
        else:
            raise ValueError('Clinical label file must contain a "Target" column or recognizable alternative.')
    return T[['ID','Target']].copy()

def fetch_targets(subject_ids, labels_map: dict) -> np.ndarray:
    """Map subject IDs to target numbers (NaN if missing)."""
    out = np.full(len(subject_ids), np.nan, dtype=float)
    for i, sid in enumerate(subject_ids):
        val = labels_map.get(str(sid))
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            out[i] = float(val)
    return out

def count_subjects(DataMap: Dict[str, Dict[str, list]]) -> int:
    """Assumes all channels have the same subject count."""
    if not DataMap:
        return 0
    first = next(iter(DataMap.values()))
    n1 = len(first['group1'])
    n2 = len(first['group2']) if isinstance(first.get('group2', []), list) else 0
    return n1 + n2
