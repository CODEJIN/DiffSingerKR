import numpy as np
import mido, os, pickle, yaml, argparse, math, librosa, hgtk, logging
from tqdm import tqdm
from pysptk.sptk import rapt
from typing import List, Tuple
from argparse import Namespace  # for type
import torch
from typing import Dict

from meldataset import mel_spectrogram, spectrogram, spec_energy
from Arg_Parser import Recursive_Parse


def AIHub_Mediazen(
    hyper_paramters: Namespace,
    dataset_path: str
    ):
    skipping_label = [line.strip() for line in open('AIHub_Mediazen_Skipping.txt', 'r').readlines()]    # This is temporal

    path_dict = {}
    for root, _, files in os.walk(os.path.join(dataset_path).replace('\\', '/')):
        for file in files:
            key, extension = os.path.splitext(file)
            if not extension.upper() in ['.WAV', '.MID']:
                continue
            if key in skipping_label:
                continue
            
            if not key in path_dict.keys():
                path_dict[key] = {}
            path_dict[key]['wav' if extension.upper() == '.WAV' else 'mid'] = os.path.join(root, file).replace('\\', '/')            

    paths = [
        (value['wav'], value['mid'], key.strip().split('_')[0].upper(), key.strip().split('_')[4].upper())
        for key, value in path_dict.items()
        if 'wav' in value.keys() and 'mid' in value.keys()
        ]
    genre_dict = {
        'BA': 'Ballade',
        'RO': 'Rock',
        'TR': 'Trot'
        }

    for index, (wav_path, midi_path, genre, singer) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'AIHub_Mediazen'
        ):
        music_label = os.path.splitext(os.path.basename(wav_path))[0]
        pattern_path = os.path.join(
            hyper_paramters.Train.Train_Pattern.Path if not index == (len(paths) - 1) else hyper_paramters.Train.Eval_Pattern.Path,
            'AIHub_Mediazen',
            singer,
            f'{music_label}.pickle'
            ).replace('\\', '/')
        if os.path.exists(pattern_path):
            continue
        genre = genre_dict[genre]

        mid = mido.MidiFile(midi_path, charset='CP949')

        music = []
        note_states = {}
        last_note = None

        # Note on 쉼표
        # From Lyric to message before note on: real note
        for message in list(mid):
            if message.type == 'note_on' and message.velocity != 0:                
                if len(note_states) == 0:
                    if message.time < 0.1:
                        music[-1][0] += message.time
                    else:
                        music.append([message.time, '<X>', 0])
                else:
                    note_states[last_note]['Time'] += message.time
                note_states[message.note] = {
                    'Lyric': None,
                    'Time': 0.0
                    }
                last_note = message.note                
            elif message.type == 'lyrics':
                if message.text == '\r' or last_note is None:    # If there is a bug in lyric
                    logging.warning(f'{wav_path} | {midi_path}')
                    continue
                note_states[last_note]['Lyric'] = message.text.strip()
                note_states[last_note]['Time'] += message.time
            elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                note_states[message.note]['Time'] += message.time
                music.append([note_states[message.note]['Time'], note_states[message.note]['Lyric'], message.note])
                del note_states[message.note]
                last_note = None
            else:
                if len(note_states) == 0:
                    music.append([message.time, '<X>', 0])
                else:
                    note_states[last_note]['Time'] += message.time
        if len(note_states) > 0:
            print(wav_path, midi_path)
            print(note_states)
            assert False
        music = [x for x in music if x[0] > 0.0]

        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)

        initial_audio_length = audio.shape[0]
        while True:
            if music[0][1] in [None, '', '<X>', 'J']:
                audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
                music = music[1:]
            else:
                break
        while True:
            if music[-1][1] in ['', '<X>', 'H']:
                music = music[:-1]
            else:
                break
        audio = audio[:int(sum([x[0] for x in music]) * hyper_paramters.Sound.Sample_Rate)]    # remove last silence
        
        # This is to avoid to use wrong data.
        if initial_audio_length * 0.5 > audio.shape[0]:
            continue

        audio = librosa.util.normalize(audio) * 0.95
        
        lyrics, notes, durations = Convert_Feature_Based_Music(
            music= music,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            frame_shift= hyper_paramters.Sound.Frame_Shift,
            consonant_duration= hyper_paramters.Duration.Consonant_Duration,
            equality_duration= hyper_paramters.Duration.Equality
            )
        lyrics_expand, notes_expand, durations_expand = Expand_by_Duration(lyrics, notes, durations)
        
        Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            duration= durations,
            lyric_expand= lyrics_expand,
            note_expand= notes_expand,
            duration_expand= durations_expand,
            audio= audio,
            music_label= music_label,
            singer= singer,
            genre= genre,
            dataset= 'AIHub_Mediazen',
            is_eval_music= index == (len(paths) - 1),
            hyper_paramters= hyper_paramters
            )

def CSD_Fix(
    hyper_paramters: Namespace,
    dataset_path: str
    ):
    paths = []
    for root, _, files in os.walk(os.path.join(dataset_path, 'wav').replace('\\', '/')):
        for file in sorted(files):
            if os.path.splitext(file)[1] != '.wav':
                continue
            wav_path = os.path.join(root, file).replace('\\', '/')
            midi_path = wav_path.replace('wav', 'mid')
            
            if not os.path.exists(midi_path):
                raise FileExistsError(midi_path)

            paths.append((wav_path, midi_path))

    for index, (wav_path, midi_path) in tqdm(
        enumerate(paths),
        total= len(paths),
        desc= 'CSD_Fix'
        ):
        music_label = os.path.splitext(os.path.basename(wav_path))[0]
        pattern_path = os.path.join(
            hyper_paramters.Train.Train_Pattern.Path if not index == (len(paths) - 1) else hyper_paramters.Train.Eval_Pattern.Path,
            'CSD',
            'CSD',
            f'{music_label}.pickle'
            ).replace('\\', '/')
        if os.path.exists(pattern_path):
            continue

        mid = mido.MidiFile(midi_path, charset='CP949')

        music = []
        note_states = {}
        last_note = None

        # Note on 쉼표
        # From Lyric to message before note on: real note        
        for message in list(mid):
            if message.type == 'note_on' and message.velocity != 0:                
                if len(note_states) == 0:
                    if message.time < 0.1:
                        music[-1][0] += message.time
                    else:
                        music.append([message.time, '<X>', 0])
                else:
                    note_states[last_note]['Time'] += message.time
                note_states[message.note] = {
                    'Lyric': None,
                    'Time': 0.0
                    }
                last_note = message.note                
            elif message.type == 'lyrics':
                if message.text == '\r':    # If there is a bug in lyric
                    logging.warning(wav_path, midi_path)
                    continue
                note_states[last_note]['Lyric'] = message.text.strip()
                note_states[last_note]['Time'] += message.time
            elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                note_states[message.note]['Time'] += message.time
                music.append([note_states[message.note]['Time'], note_states[message.note]['Lyric'], message.note])
                del note_states[message.note]
                last_note = None
            else:
                if len(note_states) == 0:
                    music.append([message.time, '<X>', 0])
                else:
                    note_states[last_note]['Time'] += message.time
        if len(note_states) > 0:
            print(wav_path, midi_path)
            print(note_states)
            assert False
        music = [x for x in music if x[0] > 0.0]

        audio, _ = librosa.load(wav_path, sr= hyper_paramters.Sound.Sample_Rate)
        
        initial_audio_length = audio.shape[0]
        while True:
            if music[0][1] in [None, '', '<X>', 'J']:
                audio = audio[int(music[0][0] * hyper_paramters.Sound.Sample_Rate):]
                music = music[1:]
            else:
                break
        while True:
            if music[-1][1] in [None, '', '<X>', 'H']:
                music = music[:-1]
            else:
                break
        audio = audio[:int(sum([x[0] for x in music]) * hyper_paramters.Sound.Sample_Rate)]    # remove last silence
        
        # This is to avoid to use wrong data.
        if initial_audio_length * 0.5 > audio.shape[0]:
            continue

        audio = librosa.util.normalize(audio) * 0.95
        
        lyrics, notes, durations = Convert_Feature_Based_Music(
            music= music,
            sample_rate= hyper_paramters.Sound.Sample_Rate,
            frame_shift= hyper_paramters.Sound.Frame_Shift,
            consonant_duration= hyper_paramters.Duration.Consonant_Duration,
            equality_duration= hyper_paramters.Duration.Equality
            )
        lyrics_expand, notes_expand, durations_expand = Expand_by_Duration(lyrics, notes, durations)

        Pattern_File_Generate(
            lyric= lyrics,
            note= notes,
            duration= durations,
            lyric_expand= lyrics_expand,
            note_expand= notes_expand,
            duration_expand= durations_expand,
            audio= audio,
            music_label= music_label,
            singer= 'CSD',
            genre= 'Children',
            dataset= 'CSD',
            is_eval_music= index == (len(paths) - 1),
            hyper_paramters= hyper_paramters
            )

def Convert_Feature_Based_Music(
    music: List[Tuple[float, str, int]],
    sample_rate: int,
    frame_shift: int,
    consonant_duration: int= 3,
    equality_duration: bool= False
    ):
    previous_used = 0
    lyrics = []
    notes = []
    durations = []
    for message_time, lyric, note in music:
        duration = round(message_time * sample_rate) + previous_used
        previous_used = duration % frame_shift
        duration = duration // frame_shift

        if lyric == '<X>':
            lyrics.append(lyric)
            notes.append(note)
            durations.append(duration)
        else:
            lyrics.extend(Decompose(lyric))
            notes.extend([note] * 3)
            if equality_duration or duration < consonant_duration * 3:
                split_duration = [duration // 3] * 3
                split_duration[1] += duration % 3
                durations.extend(split_duration)
            else:
                durations.extend([
                    consonant_duration,    # onset
                    duration - consonant_duration * 2, # nucleus
                    consonant_duration # coda
                    ])

    return lyrics, notes, durations

def Expand_by_Duration(
    lyrics: List[str],
    notes: List[int],
    durations: List[int],
    ):
    lyrics = sum([[lyric] * duration for lyric, duration in zip(lyrics, durations)], [])
    notes = sum([*[[note] * duration for note, duration in zip(notes, durations)]], [])
    durations = [index for duration in durations for index in range(duration)]

    return lyrics, notes, durations

def Decompose(syllable: str):
    onset, nucleus, coda = hgtk.letter.decompose(syllable)
    coda += '_'

    return onset, nucleus, coda

def Pattern_File_Generate(
    lyric: List[str],
    note: List[int],
    duration: List[int],
    lyric_expand: List[str],
    note_expand: List[int],
    duration_expand: List[int],
    audio: np.array,
    singer: str,
    genre: str,
    dataset: str,
    music_label: str,
    is_eval_music: bool,
    hyper_paramters: Namespace,
    ):
    spect = spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= hyper_paramters.Sound.N_FFT,
        hop_size= hyper_paramters.Sound.Frame_Shift,
        win_size= hyper_paramters.Sound.Frame_Length,
        center= False
        ).squeeze(0).T.numpy()
    mel = mel_spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= hyper_paramters.Sound.N_FFT,
        num_mels= hyper_paramters.Sound.Mel_Dim,
        sampling_rate= hyper_paramters.Sound.Sample_Rate,
        hop_size= hyper_paramters.Sound.Frame_Shift,
        win_size= hyper_paramters.Sound.Frame_Length,
        fmin= hyper_paramters.Sound.Mel_F_Min,
        fmax= hyper_paramters.Sound.Mel_F_Max,
        center= False
        ).squeeze(0).T.numpy()

    log_f0 = rapt(
        x= audio * 32768,
        fs= hyper_paramters.Sound.Sample_Rate,
        hopsize= hyper_paramters.Sound.Frame_Shift,
        min= hyper_paramters.Sound.F0_Min,
        max= hyper_paramters.Sound.F0_Max,
        otype= 2,   # log
        )[:mel.shape[0]]

    log_energy = spec_energy(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= hyper_paramters.Sound.N_FFT,
        hop_size= hyper_paramters.Sound.Frame_Shift,
        win_size=hyper_paramters.Sound.Frame_Length,
        center= False
        ).squeeze(0).log().numpy()

    if mel.shape[0] > len(lyric_expand):
        # print('Case1')
        spect = spect[math.floor((spect.shape[0] - len(lyric_expand)) / 2.0):-math.ceil((spect.shape[0] - len(lyric_expand)) / 2.0)]
        mel = mel[math.floor((mel.shape[0] - len(lyric_expand)) / 2.0):-math.ceil((mel.shape[0] - len(lyric_expand)) / 2.0)]
        log_f0 = log_f0[math.floor((log_f0.shape[0] - len(lyric_expand)) / 2.0):-math.ceil((log_f0.shape[0] - len(lyric_expand)) / 2.0)]
        log_energy = log_energy[math.floor((log_energy.shape[0] - len(lyric_expand)) / 2.0):-math.ceil((log_energy.shape[0] - len(lyric_expand)) / 2.0)]
    elif len(lyric_expand) > mel.shape[0]:
        # print('Case2')
        fix_length = len(lyric_expand) - mel.shape[0]
        if hyper_paramters.Duration.Equality:
            if duration[-1] > fix_length:
                duration[-1] = duration[-1] - fix_length
                lyric_expand = lyric_expand[:-fix_length]
                note_expand = note_expand[:-fix_length]
                duration_expand = duration_expand[:-fix_length]
            else:
                logging.warning(f'\'{dataset}-{singer}-{music_label}\' is skipped because the audio and midi length incompatible.')
                return
        else:
            consonant_duration = hyper_paramters.Duration.Consonant_Duration
            if duration[-2] > fix_length:
                duration[-2] = duration[-2] - fix_length
                lyric_expand = lyric_expand[0:-consonant_duration - fix_length] + lyric_expand[-consonant_duration:]
                note_expand = note_expand[0:-consonant_duration - fix_length] + note_expand[-consonant_duration:]
                duration_expand = duration_expand[0:-consonant_duration - fix_length] + duration_expand[-consonant_duration:]            
            else:
                logging.warning(f'\'{dataset}-{singer}-{music_label}\' is skipped because the audio and midi length incompatible.')
                return
        
    pattern = {
        'Audio': audio.astype(np.float32),
        'Spectrogram': spect.astype(np.float32),
        'Mel': mel.astype(np.float32),
        'Log_F0': log_f0.astype(np.float32),
        'Log_Energy': log_energy.astype(np.float32),
        'Lyric': lyric,
        'Note': note,
        'Duration': duration,
        'Lyric_Expand': lyric_expand,
        'Note_Expand': note_expand,
        'Duration_Expand': duration_expand,
        'Singer': singer,
        'Genre': genre,
        'Dataset': dataset,
        }

    pattern_path = os.path.join(
        hyper_paramters.Train.Train_Pattern.Path if not is_eval_music else hyper_paramters.Train.Eval_Pattern.Path,
        dataset,
        singer,
        f'{music_label}.pickle'
        ).replace('\\', '/')

    os.makedirs(os.path.dirname(pattern_path), exist_ok= True)
    pickle.dump(
        pattern,
        open(pattern_path, 'wb'),
        protocol= 4
        )

def Note_Predictor(log_f0):
    '''
    log_f0: [F0_t]
    '''
    notes = np.arange(0, 128)
    log_f0s = np.log(440 * 2 ** ((notes - 69 - 12) / 12))
    log_f0s[0] = 0.0
    criterion = np.expand_dims(log_f0s, axis= 0)   # [1, 128]

    return np.argmin(
        np.abs(np.expand_dims(log_f0, axis= 1) - criterion),
        axis= 1
        )

def Token_Dict_Generate(hyper_parameters: Namespace):
    tokens = \
        list(hgtk.letter.CHO) + \
        list(hgtk.letter.JOONG) + \
        ['{}_'.format(x) for x in hgtk.letter.JONG]
    
    os.makedirs(os.path.dirname(hyper_parameters.Token_Path), exist_ok= True)
    yaml.dump(
        {token: index for index, token in enumerate(['<S>', '<E>', '<X>'] + sorted(tokens))},
        open(hyper_parameters.Token_Path, 'w')
        )

def Metadata_Generate(
    hyper_parameters: Namespace,
    eval: bool= False
    ):
    pattern_path = hyper_parameters.Train.Eval_Pattern.Path if eval else hyper_parameters.Train.Train_Pattern.Path
    metadata_file = hyper_parameters.Train.Eval_Pattern.Metadata_File if eval else hyper_parameters.Train.Train_Pattern.Metadata_File

    spectrogram_range_dict = {}
    mel_range_dict = {}
    log_f0_dict = {}
    log_energy_dict = {}
    singers = []
    genres = []
    min_note, max_note = math.inf, -math.inf

    new_metadata_dict = {
        'N_FFT': hyper_parameters.Sound.N_FFT,
        'Mel_Dim': hyper_parameters.Sound.Mel_Dim,
        'Frame_Shift': hyper_parameters.Sound.Frame_Shift,
        'Frame_Length': hyper_parameters.Sound.Frame_Length,
        'Sample_Rate': hyper_parameters.Sound.Sample_Rate,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Feature_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Log_F0_Length_Dict': {},
        'Energy_Length_Dict': {},
        'Lyric_Length_Dict': {},
        'Note_Length_Dict': {},
        'Duration_Length_Dict': {},
        'Lyric_Expand_Length_Dict': {},
        'Note_Expand_Length_Dict': {},
        'Duration_Expand_Length_Dict': {},
        'Singer_Dict': {},
        'Genre_Dict': {},
        'File_List_by_Singer_Dict': {},
        }

    files_tqdm = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path, followlinks= True)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path, followlinks= True):
        for file in files:
            with open(os.path.join(root, file).replace('\\', '/'), 'rb') as f:
                pattern_dict = pickle.load(f)
            file = os.path.join(root, file).replace('\\', '/').replace(pattern_path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Audio', 'Spectrogram', 'Mel', 'Log_F0', 'Log_Energy', 'Lyric', 'Note', 'Singer', 'Genre', 'Dataset')
                    ]):
                    continue
                new_metadata_dict['Audio_Length_Dict'][file] = pattern_dict['Audio'].shape[0]
                new_metadata_dict['Feature_Length_Dict'][file] = pattern_dict['Spectrogram'].shape[0]
                new_metadata_dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[0]
                new_metadata_dict['Log_F0_Length_Dict'][file] = pattern_dict['Log_F0'].shape[0]
                new_metadata_dict['Energy_Length_Dict'][file] = pattern_dict['Log_Energy'].shape[0]
                new_metadata_dict['Lyric_Length_Dict'][file] = len(pattern_dict['Lyric'])
                new_metadata_dict['Note_Length_Dict'][file] = len(pattern_dict['Note'])
                new_metadata_dict['Duration_Length_Dict'][file] = len(pattern_dict['Duration'])
                new_metadata_dict['Lyric_Expand_Length_Dict'][file] = len(pattern_dict['Lyric_Expand'])
                new_metadata_dict['Note_Expand_Length_Dict'][file] = len(pattern_dict['Note_Expand'])
                new_metadata_dict['Duration_Expand_Length_Dict'][file] = len(pattern_dict['Duration_Expand'])
                new_metadata_dict['Singer_Dict'][file] = pattern_dict['Singer']
                new_metadata_dict['File_List'].append(file)
                if not pattern_dict['Singer'] in new_metadata_dict['File_List_by_Singer_Dict'].keys():
                    new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']] = []
                new_metadata_dict['File_List_by_Singer_Dict'][pattern_dict['Singer']].append(file)

                if not pattern_dict['Singer'] in spectrogram_range_dict.keys():
                    spectrogram_range_dict[pattern_dict['Singer']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Singer'] in mel_range_dict.keys():
                    mel_range_dict[pattern_dict['Singer']] = {'Min': math.inf, 'Max': -math.inf}
                if not pattern_dict['Singer'] in log_f0_dict.keys():
                    log_f0_dict[pattern_dict['Singer']] = []
                if not pattern_dict['Singer'] in log_energy_dict.keys():
                    log_energy_dict[pattern_dict['Singer']] = []
                
                spectrogram_range_dict[pattern_dict['Singer']]['Min'] = min(spectrogram_range_dict[pattern_dict['Singer']]['Min'], pattern_dict['Spectrogram'].min().item())
                spectrogram_range_dict[pattern_dict['Singer']]['Max'] = max(spectrogram_range_dict[pattern_dict['Singer']]['Max'], pattern_dict['Spectrogram'].max().item())
                mel_range_dict[pattern_dict['Singer']]['Min'] = min(mel_range_dict[pattern_dict['Singer']]['Min'], pattern_dict['Mel'].min().item())
                mel_range_dict[pattern_dict['Singer']]['Max'] = max(mel_range_dict[pattern_dict['Singer']]['Max'], pattern_dict['Mel'].max().item())
                
                log_f0_dict[pattern_dict['Singer']].append(pattern_dict['Log_F0'])
                log_energy_dict[pattern_dict['Singer']].append(pattern_dict['Log_Energy'])
                singers.append(pattern_dict['Singer'])
                genres.append(pattern_dict['Genre'])

                min_note = min(min_note, *[x for x in pattern_dict['Note'] if x > 0])
                max_note = max(max_note, *[x for x in pattern_dict['Note'] if x > 0])
            except Exception as e:
                print('File \'{}\' is not correct pattern file. This file is ignored. Error: {}'.format(file, e))
            files_tqdm.update(1)

    new_metadata_dict['Min_Note'] = min_note
    new_metadata_dict['Max_Note'] = max_note

    with open(os.path.join(pattern_path, metadata_file.upper()).replace('\\', '/'), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    if not eval:
        yaml.dump(
            spectrogram_range_dict,
            open(hp.Spectrogram_Range_Info_Path, 'w')
            )
        yaml.dump(
            mel_range_dict,
            open(hp.Mel_Range_Info_Path, 'w')
            )
        
        log_f0_info_dict = {}
        for singer, log_f0_list in log_f0_dict.items():
            log_f0 = np.hstack(log_f0_list)
            log_f0 = np.clip(log_f0, -10.0, np.inf)
            log_f0 = log_f0[log_f0 != -10.0]

            log_f0_info_dict[singer] = {
                'Mean': log_f0.mean().item(),
                'Std': log_f0.std().item(),
                }
        yaml.dump(
            log_f0_info_dict,
            open(hp.Log_F0_Info_Path, 'w')
            )

        log_energy_info_dict = {}
        for singer, log_energy_list in log_energy_dict.items():
            log_energy = np.hstack(log_energy_list)            
            log_energy_info_dict[singer] = {
                'Mean': log_energy.mean().item(),
                'Std': log_energy.std().item(),
                }
        yaml.dump(
            log_energy_info_dict,
            open(hp.Log_Energy_Info_Path, 'w')
            )

        singer_index_dict = {
            singer: index
            for index, singer in enumerate(sorted(set(singers)))
            }
        yaml.dump(
            singer_index_dict,
            open(hyper_parameters.Singer_Info_Path, 'w')
            )

        genre_index_dict = {
            genre: index
            for index, genre in enumerate(sorted(set(genres)))
            }
        yaml.dump(
            genre_index_dict,
            open(hyper_parameters.Genre_Info_Path, 'w')
            )

    print('Metadata generate done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-csd', '--csd_path', required= False)
    argparser.add_argument('-am', '--aihub_mediazen_path', required= False)
    argparser.add_argument('-hp', '--hyper_paramters', required= True)
    args = argparser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_paramters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    Token_Dict_Generate(hyper_parameters= hp)
    if args.csd_path:
        CSD_Fix(
            hyper_paramters= hp,
            dataset_path= args.csd_path
            )
    if args.aihub_mediazen_path:
        AIHub_Mediazen(
            hyper_paramters= hp,
            dataset_path= args.aihub_mediazen_path
            )
    
    Metadata_Generate(hp, False)
    Metadata_Generate(hp, True)

# python Pattern_Generator.py -hp Hyper_Parameters/Hyper_Parameters_Temp.yaml -csd D:/Datasets/rawdata_music/CSD

# python Pattern_Generator.py -hp Hyper_Parameters/Hyper_Parameters_Temp.yaml -am D:/Datasets/rawdata_music/AIHub
