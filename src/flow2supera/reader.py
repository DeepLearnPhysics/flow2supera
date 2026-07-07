import h5py 
import h5flow
import numpy as np
from yaml import Loader
import yaml
import os
import flow2supera
import time
import tqdm
import sys

FLOAT_MAX = sys.float_info.max
INT_MAX = (1 << 31) - 1

class InputEvent:
    event_id = -1
    trig_type = INT_MAX
    true_event_id = -1
    segments = None
    hits = None
    ext_trigs = None
    backtracked_hits = None
    calib_final_hits  = None
    trajectories = None
    interactions = []
    t0 = -1
    segment_index_min = -1
    event_separator = ''
    flashes = []
    light_events = []

class Flash:
    flash_id = -1
    time = FLOAT_MAX
    timeWidth = FLOAT_MAX
    PEPerOpDet = []
    volume_id = -1

class Neutrino:
    idx = INT_MAX
    interaction_id = INT_MAX
    target = INT_MAX
    x = FLOAT_MAX
    y = FLOAT_MAX
    z = FLOAT_MAX
    time = FLOAT_MAX
    pdg_code =  INT_MAX
    lepton_pdg_code = INT_MAX
    energy_init = FLOAT_MAX
    theta = FLOAT_MAX
    momentum_transfer =  FLOAT_MAX
    momentum_transfer_mag =  FLOAT_MAX
    energy_transfer =  FLOAT_MAX
    bjorken_x = FLOAT_MAX
    inelasticity = FLOAT_MAX
    px = FLOAT_MAX
    py = FLOAT_MAX
    pz = FLOAT_MAX
    lepton_p = FLOAT_MAX
    current_type = -1
    interaction_mode = -1
    interaction_type = -1
    
class InputReader:
    
    def __init__(self, config=None):

        self._event_ids = None
        self._event_t0s = None
        self._hits = None
        self._backtracked_hits = None
        self._segments = None
        self._trajectories = None
        self._interactions = None
        self._is_sim = False
        self._is_mpvmpr= False
        self._has_light=False
        self._hits_type = 'prompt'
        self._include_disabled_channels = False
        self._beam_trigger = None
        self._require_ext_trigs = False
        if config:
            if os.path.isfile(config):
                file=config
            else:
                file=flow2supera.config.get_config(config)
            with open(file,'r') as f:
                cfg=yaml.safe_load(f.read())
                if 'Flow2Supera' in cfg.keys():
                    if not isinstance(cfg['Flow2Supera'],dict):
                        raise TypeError('Flow2Supera configuration block should be a dict type')

                    if 'DataType' in cfg['Flow2Supera']:
                        self._is_sim=cfg['Flow2Supera'].get('DataType')[0]=='sim'
                        self._is_mpvmpr=cfg['Flow2Supera'].get('DataType')[1]=='mpvmpr'  
                    
                    if 'HitsType' in cfg['Flow2Supera']:
                        self._hits_type=cfg['Flow2Supera'].get('HitsType')
                        if self._hits_type != 'prompt' and self._hits_type != 'final':
                            raise ValueError('ERROR! HitsType config parameter can only be prompt or final')
                    if 'DisabledChannels' in cfg['Flow2Supera']:
                        self._include_disabled_channels=cfg['Flow2Supera'].get('DisabledChannels', self._include_disabled_channels)
                    if 'BeamTriggerIOgroup' in cfg['Flow2Supera']:
                        self._beam_trigger=cfg['Flow2Supera'].get('BeamTriggerIOgroup')
                    if 'RequireExtTrig' in cfg['Flow2Supera']:
                        self._require_ext_trigs=cfg['Flow2Supera'].get('RequireExtTrig')
        print(f'[InputReader] is sim? {self._is_sim} is mpvmpr? {self._is_mpvmpr}')
        print(f'[InputReader] Type of calibrated hits used: {self._hits_type}')

    def __len__(self):
        if self._event_ids is None: return 0
        return len(self._event_ids)

    
    def __iter__(self):
        for entry in range(len(self)):
            yield self.GetEvent(entry)

    def ReadFile(self, input_file, entries_to_read=None, verbose=False):
        if not isinstance(input_file, str):
            raise TypeError('Input file must be a str type')

        print('Reading input file...')

        # H5Flow's H5FlowDataManager class associates datasets through references.
        # Dataset-level paths (without the /data suffix) are used to dereference
        # those associations directly with the flow manager.
        self.charge_event_path = 'charge/events'
        self.calib_hits_path   = f'charge/calib_{self._hits_type}_hits'
        self.ext_trigs_path    = 'charge/ext_trigs'
        self.light_event_path  = 'light/events'
        self.light_flash_path  = 'light/flash'

        events_path            = 'charge/events/'
        calib_hits_path        = f'charge/calib_{self._hits_type}_hits/data'
        backtracked_hits_path  = f'mc_truth/calib_{self._hits_type}_hit_backtrack/data'
        interactions_path      = 'mc_truth/interactions/data'
        segments_path          = 'mc_truth/segments/data'
        trajectories_path      = 'mc_truth/trajectories/data'

        # TODO Currently only reading one input file at a time. Is it
        # necessary to read multiple? If so, how to handle non-unique
        # event IDs?
        flow_manager = h5flow.data.H5FlowDataManager(input_file, 'r')
        self.manager = flow_manager
        with h5py.File(input_file, 'r') as fin:
            events = flow_manager[events_path]
            events_data = events['data']
            self._event_ids = events_data['id']
            self._event_t0s = events_data['unix_ts'] + events_data['unix_ts_usec']/1e6
            self._hits      = flow_manager[calib_hits_path]
            self._has_ext_trigs = 'ext_trigs' in fin['charge'].keys()
            self._n_ext_trigs   = events_data['n_ext_trigs']
            if self._hits is None:
                raise ValueError(f'No data in {calib_hits_path}')
            if entries_to_read is not None:
                self._event_ids = events_data['id'][:entries_to_read]
            if self._include_disabled_channels and 'is_disabled' not in self._hits.dtype.names:
                raise ValueError ('No disabled channels field in hits dataset, please change config')
            self._has_light = 'light' in fin.keys() and 'flash' in fin['light'].keys()

            if self._is_sim:
                self._backtracked_hits  = flow_manager[backtracked_hits_path]
                if self._backtracked_hits is None:
                    raise ValueError(f'No data in {backtracked_hits_path}')
                self._segments     = np.array(flow_manager[segments_path])
                self._trajectories = np.array(flow_manager[trajectories_path])
                self._interactions = np.array(flow_manager[interactions_path])

                # Make explicit reference to segment ids and entry index array
                self._segment_ids = self._segments['segment_id']
                self._segment_idx = np.arange(len(self._segments))
                self._segment_event_ids = self._segments['event_id']

                self._valid_segment_event_ids = self.FileQualityCheck(entries_to_read)

    def DereferenceEvent(self, entry, dataset_path):
        '''
        Fetch the data associated with a charge event (row index = entry) by
        dereferencing the charge/events -> dataset_path references with the
        flow manager. Returns a plain structured array with masked (invalid)
        elements removed.
        '''
        data = self.manager[self.charge_event_path, dataset_path, entry]
        data = data.flatten()
        valid = ~np.ma.getmaskarray(data[data.dtype.names[0]])
        return data.data[valid]

    def GetEventHitIndices(self, entry):
        '''
        Row indices (sorted) of the calibrated hits associated with a charge
        event, obtained by dereferencing the flow manager references. The same
        indices are valid for the backtracking dataset, which is row-aligned
        with the calibrated hits.
        '''
        ref, ref_dir = self.manager.get_ref(self.charge_event_path, self.calib_hits_path)
        region = self.manager.get_ref_region(self.charge_event_path, self.calib_hits_path)
        idx = h5flow.data.dereference(entry, ref, region=region, ref_direction=ref_dir, indices_only=True)
        idx = idx.flatten()
        idx = idx.data[~np.ma.getmaskarray(idx)]
        return np.sort(idx)

    @staticmethod
    def ReadDatasetRows(dataset, indices):
        '''Read the given (sorted) rows of a h5py dataset, using a contiguous slice when possible'''
        if len(indices) == 0:
            return np.empty(0, dtype=dataset.dtype)
        if indices[-1] - indices[0] + 1 == len(indices):
            return dataset[indices[0]:indices[-1]+1]
        return dataset[indices]

    
    def GetNeutrinoIxn(self, ixn, ixn_idx):

        interaction = Neutrino()
        
        # if isinstance(ixn,np.void):
        #     return interaction
        
        interaction.idx = int(ixn_idx)
        interaction.interaction_id = int(ixn['vertex_id']) 
        interaction.target = int(ixn['target'])
        try:
            interaction.x = ixn['vertex'][0]
            interaction.y = ixn['vertex'][1]
            interaction.z = ixn['vertex'][2]
            interaction.time = ixn['vertex'][3]
        except ValueError:
            interaction.x = ixn['x_vert']
            interaction.y = ixn['y_vert']
            interaction.z = ixn['z_vert']
            interaction.time = ixn['t_vert']
        interaction.pdg_code = int(ixn['nu_pdg'])
        interaction.lepton_pdg_code = int(ixn['lep_pdg'])  
        interaction.energy_init = ixn['Enu']
        interaction.theta = ixn['lep_ang']
        interaction.momentum_transfer =  ixn['Q2']
        interaction.momentum_transfer_mag =  ixn['q3']
        interaction.energy_transfer =  ixn['q0']
        interaction.bjorken_x = ixn['x']
        interaction.inelasticity = ixn['y']
        interaction.px = ixn['nu_4mom'][0]
        interaction.py = ixn['nu_4mom'][1]       
        interaction.pz = ixn['nu_4mom'][2]
        interaction.lepton_p = ixn['lep_mom']
        if(ixn['isCC']): interaction.current_type = 0
        else: interaction.current_type = 1
        interaction.interaction_mode = int(ixn['reaction'])
        interaction.interaction_type = int(ixn['reaction'])   
        
        return interaction  
    
    def GetFlash(self, flash, t0):

        flash_result = Flash()
        flash_result.flash_id = int(flash['id'])
        flash_result.time = flash['hit_time_range'][0]*1e-9 + t0
        flash_result.timeWidth = (flash['hit_time_range'][1] - flash['hit_time_range'][0])*1e-9
        flash_result.PEPerOpDet = np.array(flash['deconv_sum']).flatten()#*0.022857 #adc charge to pe conversion
        flash_result.volume_id = int(flash['tpc'])

        return flash_result
    
    # To truth associations go as hits -> segments -> trajectories


    def GetEventIDFromSegments(self, backtracked_hits):
        
        try:

            seg_ids = np.unique(np.concatenate([bhit['segment_ids'][bhit['fraction']!=0.] for bhit in backtracked_hits]))
            
            sid_min,sid_max = seg_ids.min(),seg_ids.max()

            seg_range_mask = (self._segment_ids >= sid_min) & (self._segment_ids <= sid_max)

            event_segs=self._segment_ids[seg_range_mask]
            event_idxs=self._segment_idx[seg_range_mask]

            seg_mask = [event_idxs[i] for i in range(len(event_segs)) if event_segs[i] in seg_ids]

            return np.unique(self._segment_event_ids[seg_mask])

        except ValueError:
            valid_frac_counts = [(bhit['fraction']!=0.).sum() for bhit in backtracked_hits]
            if sum(valid_frac_counts) > 0:
                # case the original error was not due to empty association, re-raise
                raise
            print(f'[InputReader] UNEXPECTED: found no hit with any association to the truth hit')
            return np.array([])


    def FileQualityCheck(self,entries_to_read=None):

        num_entries = len(self._event_ids)
        if entries_to_read is not None:
            num_entries = min(num_entries,int(entries_to_read))
        eid_ctr = np.zeros(num_entries,dtype=int)
        eid_val = np.full(num_entries,fill_value=-1,dtype=int)
        bad_event_ids = []
        empty_entries = []

        print(f'[InputReader] Checking the event IDs in this file... (reading {num_entries})')
        for entry in tqdm.tqdm(range(num_entries),desc='Scanning event IDs'):

            bhits = self.ReadDatasetRows(self._backtracked_hits, self.GetEventHitIndices(entry))
            if len(bhits) == 0:
                empty_entries.append(entry)
                continue
            ids_this=self.GetEventIDFromSegments(bhits)
            if not len(ids_this) == 1:
                eid_ctr[entry] = len(ids_this)
                if len(ids_this):
                    bad_event_ids.append(ids_this)
            else:
                eid_val[entry] = ids_this[0]

        bad_entries = (eid_val == -1).nonzero()[0]
        
        # Filter out bad entries that are in empty_entries
        bad_entries = [entry for entry in bad_entries if entry not in empty_entries]
        
        if len(empty_entries) > 0:
            print('[Inputreader] WARNING: These entries have no hits:', empty_entries)
 
        if len(bad_entries) > 0:
            print('[Inputreader] WARNING: entries where more than one event ID is found:',bad_entries)
            print('                       corresponding event IDs stored:',[list(ids) for ids in bad_event_ids])

        # Find other impacted entries
        if len(bad_event_ids):
            bad_event_ids=np.concatenate(bad_event_ids)
        bad_event_ids = np.unique(bad_event_ids)
        mask=np.zeros(num_entries,dtype=bool)
        for bad_id in bad_event_ids:
            mask = mask | (eid_val == bad_id)
        if mask.sum():
            print('[Inputreader] WARNING: other entries impacted by bad event IDs:',mask.nonzero()[0])

        entry_mask = mask | (eid_val == -1)
        eid_val[entry_mask] = -1

        # if bad_entries:
        #     raise ValueError("ERROR: Terminating due to multiple true event id association. Check simulation")
            
        return eid_val
        

    def GetEntry(self, entry):
        
        if entry >= len(self._event_ids):
            print('[Inputreader] Entry {} is above allowed entry index ({})'.format(entry, len(self._event_ids)))
            print('              Invalid read request (returning None)')
            return None

        if self._require_ext_trigs and self._n_ext_trigs[entry] == 0:
            print(f'Skipping event {entry} as there is no external triggers but the configuration requires it.')
            return None

        t0=time.time()
        result = InputEvent()

        result.event_id = self._event_ids[entry]

        result.t0 = self._event_t0s[entry]

        hit_indices = self.GetEventHitIndices(entry)
        all_hits = self.ReadDatasetRows(self._hits, hit_indices)

        if self._include_disabled_channels:
            disabled_mask = ~all_hits['is_disabled']
            result.hits = all_hits[disabled_mask]
        else:
            result.hits = all_hits

        #TODO: determine a hit threshold for noisy events
        # if not self._is_sim and len(result.hits) < 50: #remove noisy events
        #     print(f'[InputReader] No hits, skipping this entry ({entry})...')
        #     return result

        if self._has_ext_trigs:
            ext_trigs = self.DereferenceEvent(entry, self.ext_trigs_path)
            if len(ext_trigs): #if there is asociated external trigger
                ttypes = ext_trigs['iogroup']
                if self._beam_trigger in ttypes: #beam takes precedence in data
                    result.trig_type = self._beam_trigger
                else: #otherwise just take the first trigger if more than 1
                    result.trig_type = ttypes[0]

        if self._has_light:
            result.light_events = []
            event_flashes = []
            light_events = self.manager[self.charge_event_path, self.light_event_path, result.event_id]
            if not np.ma.is_masked(light_events['id']): 
                result.light_events = light_events.flatten()
                for lev in result.light_events:
                    flashes = self.manager[self.light_event_path, self.light_flash_path, lev['id']]
                    if not np.ma.is_masked(flashes['id']): event_flashes.extend(flashes.flatten())


            result.flashes = []
            for flash in event_flashes:
                flash_result = self.GetFlash(flash, result.t0) 
                result.flashes.append(flash_result)
                
        if not self._is_sim:
            print('[InputReader] SuperaInput filled (not sim)',time.time()-t0,'[s]')
            return result

        # The backtracking dataset is row-aligned with the calibrated hits,
        # so the same row indices apply
        all_backtracked_hits = self.ReadDatasetRows(self._backtracked_hits, hit_indices)
        if self._include_disabled_channels:
            result.backtracked_hits = all_backtracked_hits[disabled_mask]
        else:
            result.backtracked_hits = all_backtracked_hits
        
        if self._valid_segment_event_ids[entry] < 0:
            print(f'[InputReader] Skipping this entry ({entry})...')
            return result
        
        st_event_id = self._valid_segment_event_ids[entry]
        
        result.segments = self._segments[self._segments['event_id']==st_event_id]
        result.trajectories = self._trajectories[self._trajectories['event_id']==st_event_id]
        
        result.true_event_id = st_event_id


        if self._is_mpvmpr:
            print('[InputReader] SuperaInput filled (sim, mpvmpr)',time.time()-t0,'[s]')
            return result
        
        #Find true neutrino interactions associated with the reco events
        result.interactions = []
        
        interactions_array  = np.array(self._interactions)
        event_interactions = interactions_array[interactions_array['event_id'] == result.true_event_id]
        for ixn_idx, ixn in enumerate(event_interactions):
            interaction = self.GetNeutrinoIxn(ixn, ixn_idx)
            result.interactions.append(interaction)  
        
        print('[InputReader] SuperaInput filled (sim, not mpvmpr)',time.time()-t0,'[s]')
        return result 


    def EventDump(self, input_event):
        print('-----------EVENT DUMP [InputReader] ------------')
        print('Event ID {}'.format(input_event.event_id))
        print('Event t0 {}'.format(input_event.t0))
        print('External trigger type {}'.format(input_event.trig_type))
        print('Hits shape:', input_event.hits.shape)

        if self._has_light:
            if input_event.light_events: print('Associated light events:', len(input_event.light_events))
            if input_event.flashes: print('Associated flashes:', len(input_event.flashes))
        
        if self._is_sim and len(input_event.hits) !=0:
            print('True event ID {}'.format(input_event.true_event_id))
            print('Backtracked hits len:', len(input_event.backtracked_hits))
            print('Segments in this event:', len(input_event.segments))
            print('Trajectories in this event:', len(input_event.trajectories))
            print('Interactions in this event:', len(input_event.interactions))


