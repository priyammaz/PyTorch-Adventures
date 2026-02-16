import torchaudio

def load_audios(path_to_audios, sampling_rate):
    if not isinstance(path_to_audios, list):
        path_to_audios = [path_to_audios]
    
    waveforms = []
    for path in path_to_audios:
        waveform, sr = torchaudio.load(path)

        # Check sampling rate
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)    
        
        # Make sure its single channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Add batch dimension 
        waveform = waveform.unsqueeze(0)

        waveforms.append(waveform)
    
    return waveforms

def save_audios(waveforms, output_paths, sampling_rate):
    
    if not isinstance(waveforms, list):
        waveforms = [waveforms]
    
    if not isinstance(output_paths, list):
        output_paths = [output_paths]
    
    if len(waveforms) != len(output_paths):
        raise ValueError(f"Number of waveforms ({len(waveforms)}) must match number of output paths ({len(output_paths)})")
    
    for waveform, path in zip(waveforms, output_paths):

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        
        torchaudio.save(path, waveform, sampling_rate)
        


