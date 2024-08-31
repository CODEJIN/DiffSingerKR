from argparse import Namespace
import torch
import numpy as np
import pickle, os, logging
from typing import Dict, List, Optional
import hgtk

from Pattern_Generator import Convert_Feature_Based_Music, Expand_by_Duration

def Decompose(syllable: str):    
    onset, nucleus, coda = hgtk.letter.decompose(syllable)
    coda += '_'

    return onset, nucleus, coda

def Lyric_to_Token(lyric: List[str], token_dict: Dict[str, int]):
    return [
        token_dict[letter]
        for letter in list(lyric)
        ]

def Token_Stack(tokens: List[List[int]], token_dict: Dict[str, int], max_length: Optional[int]= None):
    max_token_length = max_length or max([len(token) for token in tokens])
    tokens = np.stack(
        [np.pad(token[:max_token_length], [0, max_token_length - len(token[:max_token_length])], constant_values= token_dict['<X>']) for token in tokens],
        axis= 0
        )
    return tokens

def Note_Stack(notes: List[List[int]], max_length: Optional[int]= None):
    max_note_length = max_length or max([len(note) for note in notes])
    notes = np.stack(
        [np.pad(note[:max_note_length], [0, max_note_length - len(note[:max_note_length])], constant_values= 0) for note in notes],
        axis= 0
        )
    return notes

def Duration_Stack(durations: List[List[int]], max_length: Optional[int]= None):
    max_duration_length = max_length or max([len(duration) for duration in durations])
    durations = np.stack(
        [np.pad(duration[:max_duration_length], [0, max_duration_length - len(duration[:max_duration_length])], constant_values= 0) for duration in durations],
        axis= 0
        )
    return durations

def Feature_Stack(features: List[np.array], max_length: Optional[int]= None):
    max_feature_length = max_length or max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_feature_length - feature.shape[0]], [0, 0]], constant_values= -1.0) for feature in features],
        axis= 0
        )
    return features

def Log_F0_Stack(log_f0s: List[np.array], max_length: int= None):
    max_log_f0_length = max_length or max([len(log_f0) for log_f0 in log_f0s])
    log_f0s = np.stack(
        [np.pad(log_f0, [0, max_log_f0_length - len(log_f0)], constant_values= 0.0) for log_f0 in log_f0s],
        axis= 0
        )
    return log_f0s


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        singer_info_dict: Dict[str, int],
        genre_info_dict: Dict[str, int],
        feature_range_info_dict: Dict[str, Dict[str, float]],
        pattern_path: str,
        metadata_file: str,
        feature_type: str,
        pattern_length: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0
        ):
        super().__init__()
        self.token_dict = token_dict
        self.singer_info_dict = singer_info_dict
        self.genre_info_dict = genre_info_dict
        self.feature_range_info_dict = feature_range_info_dict
        self.pattern_path = pattern_path
        self.feature_type = feature_type

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))

        self.patterns = []
        max_pattern_by_singer = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Singer_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Singer_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_singer)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns = [
            x for x in self.patterns
            if metadata_dict['Lyric_Expand_Length_Dict'][x] >= pattern_length
            ] * accumulated_dataset_epoch
        self.true_pattern_length = len(self.patterns) // accumulated_dataset_epoch

        self.pattern_dict = {}

    def __getitem__(self, idx):
        if idx % self.true_pattern_length in self.pattern_dict.keys():
            return self.pattern_dict[idx % self.true_pattern_length]
        
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        pattern_dict = pickle.load(open(path, 'rb'))

        feature = pattern_dict[self.feature_type]
        feature_min = self.feature_range_info_dict[pattern_dict['Singer']]['Min']
        feature_max = self.feature_range_info_dict[pattern_dict['Singer']]['Max']
        feature = (pattern_dict[self.feature_type] - feature_min) / (feature_max - feature_min) * 2.0 - 1.0

        singer = self.singer_info_dict[pattern_dict['Singer']]
        genre = self.genre_info_dict[pattern_dict['Genre']]

        self.pattern_dict[idx % self.true_pattern_length] = Lyric_to_Token(pattern_dict['Lyric_Expand'], self.token_dict), pattern_dict['Note_Expand'], pattern_dict['Duration_Expand'], singer, genre, feature

        return self.pattern_dict[idx % self.true_pattern_length]

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],        
        singer_info_dict: Dict[str, int],
        genre_info_dict: Dict[str, int],
        durations: List[List[float]],
        lyrics: List[List[str]],
        notes: List[List[int]],
        singers: List[str],
        genres: List[str],
        sample_rate: int,
        frame_shift: int,
        equality_duration: bool= False,
        consonant_duration: int= 3
        ):
        super().__init__()
        self.token_dict = token_dict
        self.singer_info_dict = singer_info_dict
        self.genre_info_dict = genre_info_dict
        self.equality_duration = equality_duration
        self.consonant_duration = consonant_duration

        self.patterns = []
        for index, (duration, lyric, note, singer, genre) in enumerate(zip(durations, lyrics, notes, singers, genres)):
            if not singer in self.singer_info_dict.keys():
                logging.warn('The singer \'{}\' is incorrect. The pattern \'{}\' is ignoired.'.format(singer, index))
                continue
            if not genre in self.genre_info_dict.keys():
                logging.warn('The genre \'{}\' is incorrect. The pattern \'{}\' is ignoired.'.format(genre, index))
                continue
                        
            music = [x for x in zip(duration, lyric, note)]            
            singer_label = singer
            text = lyric
            
            lyric, note, duration = Convert_Feature_Based_Music(
                music= music,
                sample_rate= sample_rate,
                frame_shift= frame_shift,
                consonant_duration= consonant_duration,
                equality_duration= equality_duration
                )
            lyric_expand, note_expand, duration_expand = Expand_by_Duration(lyric, note, duration)

            singer = self.singer_info_dict[singer]
            genre = self.genre_info_dict[genre]

            self.patterns.append((lyric_expand, note_expand, duration_expand, singer, genre, singer_label, text))

    def __getitem__(self, idx):
        lyric, note, duration, singer, genre, singer_label, text = self.patterns[idx]

        return Lyric_to_Token(lyric, self.token_dict), note, duration, singer, genre, singer_label, text

    def __len__(self):
        return len(self.patterns)

class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int],
        pattern_length: int
        ):
        self.token_dict = token_dict
        self.pattern_length = pattern_length

    def __call__(self, batch):
        tokens, notes, durations, singers, genres, features = zip(*batch)        
        
        offsets = []
        for token in tokens:
            offset = np.random.randint(0, len(token) - self.pattern_length + 1)
            offsets.append(offset)
        
        tokens = Token_Stack([
            token[offset:offset+self.pattern_length]
            for token, offset in zip(tokens, offsets)
            ], self.token_dict)
        notes = Note_Stack([
            note[offset:offset+self.pattern_length]
            for note, offset in zip(notes, offsets)
            ])
        durations = Duration_Stack([
            duration[offset:offset+self.pattern_length]
            for duration, offset in zip(durations, offsets)
            ])
        features = Feature_Stack([
            feature[offset:offset+self.pattern_length]
            for feature, offset in zip(features, offsets)
            ])

        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        notes = torch.LongTensor(notes) # [Batch, Token_t]
        durations = torch.LongTensor(durations) # [Batch, Token_t]
        lengths = torch.LongTensor([self.pattern_length] * len(batch))
        singers = torch.LongTensor(singers)  # [Batch]
        genres = torch.LongTensor(genres)  # [Batch]
        features = torch.FloatTensor(features).permute(0, 2, 1)   # [Batch, Feature_d, Feature_t]

        return tokens, notes, durations, lengths, singers, genres, features

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int]
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, notes, durations, singers, genres, singer_labels, lyrics = zip(*batch)
        
        lengths = np.array([len(token) for token in tokens])

        max_length = max(lengths)

        tokens = Token_Stack(tokens, self.token_dict, max_length)
        notes = Note_Stack(notes, max_length)        
        durations = Duration_Stack(durations, max_length)

        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        notes = torch.LongTensor(notes)   # [Batch, Time]
        durations = torch.LongTensor(durations)   # [Batch, Time]
        lengths = torch.LongTensor(lengths)   # [Batch]
        singers = torch.LongTensor(singers)  # [Batch]
        genres = torch.LongTensor(genres)  # [Batch]
        
        lyrics = [''.join([(x if x != '<X>' else ' ') for x in lyric]) for lyric in lyrics]

        return tokens, notes, durations, lengths, singers, genres, singer_labels, lyrics