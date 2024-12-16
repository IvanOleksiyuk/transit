import pandas as pd
import numpy as np

LHCO_path = "/srv/beegfs/scratch/groups/rodem/LHCO/event_anomalydetection_v2.h5"
m_jj_path = "/srv/beegfs/scratch/groups/rodem/LHCO/lhco_event_masses.h5"
n_bkg= None
n_sig= 2000
n_csts= 200
mjj_window= [[2700, 3300], [3700, 6000]]
events = pd.read_hdf("/srv/beegfs/scratch/groups/rodem/LHCO/event_anomalydetection_v2.h5")

is_signal = events.iloc[:, -1].to_numpy()
events_all = events.to_numpy()[:, :2100].reshape(-1, 700, 3)

# Simple energy rescaling
events_all[:, :, 0] = np.sqrt(events_all[:, :, 0])
#select upt to 400 constituents per event
events_all = events_all[:, :n_csts, :]

masses = pd.read_hdf(m_jj_path)
mjj = masses.to_numpy()

# Select correct signal doping 
events_bkg = events_all[is_signal == 0]
events_sig = events_all[is_signal == 1]
mjj_bkg = mjj[is_signal == 0]
mjj_sig = mjj[is_signal == 1]
events = np.concatenate([events_bkg[:n_bkg], events_sig[:n_sig]], axis=0)
masses = np.concatenate([mjj_bkg[:n_bkg], mjj_sig[n_bkg:n_bkg+n_sig]], axis=0)

lowerSB_ids = np.where((masses > mjj_window[0][0]) & (masses < mjj_window[0][1]))[0]
higherSB_ids = np.where((masses > mjj_window[1][0]) & (masses < mjj_window[1][1]))[0]
events = np.concatenate([events[lowerSB_ids], events[higherSB_ids]], axis=0)
masses = np.concatenate([masses[lowerSB_ids], masses[higherSB_ids]], axis=0)