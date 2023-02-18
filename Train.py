import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, wandb
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from librosa import griffinlim
from scipy.io import wavfile

from Modules.Modules import DiffSinger
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Noam_Scheduler import Noam_Scheduler
from Logger import Logger

from meldataset import spectral_de_normalize_torch
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.gpu_id == 0:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }
            
            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp)
                    )
                wandb.watch(self.model_dict['DiffSinger'])

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path), Loader=yaml.Loader)
        singer_info_dict = yaml.load(open(self.hp.Singer_Info_Path), Loader=yaml.Loader)
        genre_info_dict = yaml.load(open(self.hp.Genre_Info_Path), Loader=yaml.Loader)

        if self.hp.Feature_Type == 'Spectrogram':
            self.feature_range_info_dict = yaml.load(open(self.hp.Spectrogram_Range_Info_Path), Loader=yaml.Loader)
        if self.hp.Feature_Type == 'Mel':
            self.feature_range_info_dict = yaml.load(open(self.hp.Mel_Range_Info_Path), Loader=yaml.Loader)

        train_dataset = Dataset(
            token_dict= token_dict,
            singer_info_dict= singer_info_dict,
            genre_info_dict= genre_info_dict,
            feature_range_info_dict= self.feature_range_info_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            pattern_length= self.hp.Train.Pattern_Length,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            singer_info_dict= singer_info_dict,
            genre_info_dict= genre_info_dict,
            feature_range_info_dict= self.feature_range_info_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            feature_type= self.hp.Feature_Type,
            pattern_length= self.hp.Train.Pattern_Length,
            accumulated_dataset_epoch= self.hp.Train.Eval_Pattern.Accumulated_Dataset_Epoch,
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            singer_info_dict= singer_info_dict,
            genre_info_dict= genre_info_dict,
            durations= self.hp.Train.Inference_in_Train.Duration,
            lyrics= self.hp.Train.Inference_in_Train.Lyric,
            notes= self.hp.Train.Inference_in_Train.Note,
            singers= self.hp.Train.Inference_in_Train.Singer,
            genres= self.hp.Train.Inference_in_Train.Genre,
            sample_rate= self.hp.Sound.Sample_Rate,
            frame_shift= self.hp.Sound.Frame_Shift,
            equality_duration= self.hp.Duration.Equality,
            consonant_duration= self.hp.Duration.Consonant_Duration
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(
            token_dict= token_dict,
            pattern_length= self.hp.Train.Pattern_Length
            )
        inference_collater = Inference_Collater(
            token_dict= token_dict
            )

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.DistributedSampler(eval_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_dict = {
            'DiffSinger': DiffSinger(self.hp).to(self.device)
            }
        self.criterion_dict = {
            'MSE': torch.nn.MSELoss().to(self.device),
            'MAE': torch.nn.L1Loss(reduce= None).to(self.device)
            }

        self.optimizer_dict = {
            'DiffSinger': torch.optim.NAdam(
                params= self.model_dict['DiffSinger'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                )
            }
        self.scheduler_dict = {
            'DiffSinger': Noam_Scheduler(
                optimizer= self.optimizer_dict['DiffSinger'],
                warmup_steps= self.hp.Train.Learning_Rate.Warmup_Step,
                )
            }

        if self.hp.Feature_Type == 'Mel':
            self.vocoder = torch.jit.load('vocoder.pts', map_location='cpu').to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        if self.gpu_id == 0:
            logging.info(self.model_dict)

    def Train_Step(self, tokens, notes, durations, lengths, singers, genres, features):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = notes.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)
        genres = genres.to(self.device, non_blocking=True)
        singers = singers.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            linear_predictions, _, noises, epsilons = self.model_dict['DiffSinger'](
                tokens= tokens,
                notes= notes,
                durations= durations,
                lengths= lengths,
                genres= genres,
                singers= singers,
                features= features
                )

            loss_dict['Linear'] = self.criterion_dict['MSE'](
                linear_predictions,
                features,
                ).mean()
            loss_dict['Diffusion'] = self.criterion_dict['MSE'](
                noises,
                epsilons,
                ).mean()

            self.optimizer_dict['DiffSinger'].zero_grad()
            self.scaler.scale(loss_dict['Linear'] + loss_dict['Diffusion']).backward()
            if self.hp.Train.Gradient_Norm > 0.0:
                self.scaler.unscale_(self.optimizer_dict['DiffSinger'])
                torch.nn.utils.clip_grad_norm_(
                    parameters= self.model_dict['DiffSinger'].parameters(),
                    max_norm= self.hp.Train.Gradient_Norm
                    )

            self.scaler.step(self.optimizer_dict['DiffSinger'])
            self.scaler.update()
            self.scheduler_dict['DiffSinger'].step()
            self.steps += 1
            self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for tokens, notes, durations, lengths, singers, genres, features in self.dataloader_dict['Train']:
            self.Train_Step(tokens, notes, durations, lengths, singers, genres, features)

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler_dict['DiffSinger'].get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    @torch.no_grad()
    def Evaluation_Step(self, tokens, notes, durations, lengths, singers, genres, features):
        loss_dict = {}
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)
        genres = genres.to(self.device, non_blocking=True)
        singers = singers.to(self.device, non_blocking=True)
        features = features.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            linear_predictions, _, noises, epsilons = self.model_dict['DiffSinger'](
                tokens= tokens,
                notes= notes,
                durations= durations, 
                lengths= lengths,
                genres= genres,
                singers= singers,
                features= features
                )

            loss_dict['Linear'] = self.criterion_dict['MSE'](
                linear_predictions,
                features,
                ).mean()
            loss_dict['Diffusion'] = self.criterion_dict['MSE'](
                noises,
                epsilons,
                ).mean()

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        for model in self.model_dict.values():
            model.eval()

        for step, (tokens, notes, durations, lengths, singers, genres, features) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            self.Evaluation_Step(tokens, notes, durations, lengths, singers, genres, features)
        
        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['DiffSinger'], 'DiffSinger', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, tokens.size(0))

            with torch.inference_mode():
                linear_predictions, diffusion_predictions, _, _ = self.model_dict['DiffSinger'](
                    tokens= tokens[index].unsqueeze(0).to(self.device),
                    notes= notes[index].unsqueeze(0).to(self.device),
                    durations= durations[index].unsqueeze(0).to(self.device),
                    lengths= lengths[index].unsqueeze(0).to(self.device),
                    genres= genres[index].unsqueeze(0).to(self.device),
                    singers= singers[index].unsqueeze(0).to(self.device),
                    ddim_steps= max(self.hp.Diffusion.Max_Step // 10, 100)
                    )
                linear_predictions = linear_predictions.clamp(-1.0, 1.0)
                diffusion_predictions = diffusion_predictions.clamp(-1.0, 1.0)

                feature_min = min([value['Min'] for value in self.feature_range_info_dict.values()])
                feature_max = max([value['Max'] for value in self.feature_range_info_dict.values()])
                target_feature = (features[index] + 1.0) / 2.0 * (feature_max - feature_min) + feature_min
                linear_prediction_feature = (linear_predictions[0] + 1.0) / 2.0 * (feature_max - feature_min) + feature_min
                diffusion_prediction_feature = (diffusion_predictions[0] + 1.0) / 2.0 * (feature_max - feature_min) + feature_min
                
                if self.hp.Feature_Type == 'Mel':
                    target_audio = self.vocoder(target_feature.unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy() / 32768.0
                    linear_prediction_audio = self.vocoder(linear_prediction_feature.unsqueeze(0)).squeeze(0).cpu().numpy() / 32768.0
                    diffusion_prediction_audio = self.vocoder(diffusion_prediction_feature.unsqueeze(0)).squeeze(0).cpu().numpy() / 32768.0
                elif self.hp.Feature_Type == 'Spectrogram':
                    target_audio = griffinlim(spectral_de_normalize_torch(target_feature.squeeze(0)).cpu().numpy())
                    linear_prediction_audio = griffinlim(spectral_de_normalize_torch(linear_prediction_feature.squeeze(0)).cpu().numpy())
                    diffusion_prediction_audio = griffinlim(spectral_de_normalize_torch(diffusion_prediction_feature.squeeze(0)).cpu().numpy())

            image_dict = {
                'Feature/Target': (target_feature.squeeze(0).cpu().numpy(), None, 'auto', None, None, None),
                'Feature/linear': (linear_prediction_feature.squeeze(0).cpu().numpy(), None, 'auto', None, None, None),
                'Feature/Diffusion': (diffusion_prediction_feature.squeeze(0).cpu().numpy(), None, 'auto', None, None, None)
                }
            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)

            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Linear': (linear_prediction_audio, self.hp.Sound.Sample_Rate),
                'Audio/Diffusion': (diffusion_prediction_audio, self.hp.Sound.Sample_Rate),
                }
            self.writer_dict['Evaluation'].add_audio_dict(audio_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {                        
                        'Evaluation.Feature.Target': wandb.Image(target_feature.squeeze(0).cpu().numpy()),
                        'Evaluation.Feature.Linear': wandb.Image(linear_prediction_feature.squeeze(0).cpu().numpy()),
                        'Evaluation.Feature.Diffusion': wandb.Image(diffusion_prediction_feature.squeeze(0).cpu().numpy()),
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Target'
                            ),
                        'Evaluation.Audio.Linear': wandb.Audio(
                            linear_prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Prediction'
                            ),
                        'Evaluation.Audio.Diffusion': wandb.Audio(
                            diffusion_prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Prediction'
                            ),
                        },
                    step= self.steps,
                    commit= False
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        for model in self.model_dict.values():
            model.train()


    @torch.inference_mode()
    def Inference_Step(self, tokens, notes, durations, lengths, singers, genres, singer_labels, lyrics, start_index= 0, tag_step= False):
        tokens = tokens.to(self.device, non_blocking=True)
        notes = notes.to(self.device, non_blocking=True)
        durations = durations.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)
        genres = genres.to(self.device, non_blocking=True)
        singers = singers.to(self.device, non_blocking=True)

        linear_predictions, diffusion_predictions, _, _ = self.model_dict['DiffSinger'](
            tokens= tokens,
            notes= notes,
            durations= durations,
            lengths= lengths,
            genres= genres,
            singers= singers,
            ddim_steps= max(self.hp.Diffusion.Max_Step // 10, 100)
            )
        linear_predictions = linear_predictions.clamp(-1.0, 1.0)
        diffusion_predictions = diffusion_predictions.clamp(-1.0, 1.0)

        linear_prediction_list, diffusion_prediction_list = [], []
        for linear_prediction, diffusion_prediction, singer in zip(linear_predictions, diffusion_predictions, singer_labels):
            feature_max = self.feature_range_info_dict[singer]['Max']
            feature_min = self.feature_range_info_dict[singer]['Min']
            linear_prediction_list.append((linear_prediction + 1.0) / 2.0 * (feature_max - feature_min) + feature_min)
            diffusion_prediction_list.append((diffusion_prediction + 1.0) / 2.0 * (feature_max - feature_min) + feature_min)
        linear_predictions = torch.stack(linear_prediction_list, dim= 0)
        diffusion_predictions = torch.stack(diffusion_prediction_list, dim= 0)
        
        if self.hp.Feature_Type == 'Mel':
            linear_audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))].cpu().numpy()
                for audio, length in zip(
                    self.vocoder(linear_predictions),
                    lengths
                    )
                ]
            diffusion_audios = [
                audio[:min(length * self.hp.Sound.Frame_Shift, audio.size(0))].cpu().numpy()
                for audio, length in zip(
                    self.vocoder(diffusion_predictions),
                    lengths
                    )
                ]
        elif self.hp.Feature_Type == 'Spectrogram':
            linear_audios, diffusion_audios = [], []
            for linear_prediction, diffusion_prediction, length in zip(
                linear_predictions,
                diffusion_predictions,
                lengths
                ):
                linear_prediction = spectral_de_normalize_torch(linear_prediction).cpu().numpy()
                linear_audio = griffinlim(linear_prediction)[:min(linear_prediction.size(1), length) * self.hp.Sound.Frame_Shift]
                linear_audio = (linear_audio / np.abs(linear_audio).max() * 32767.5).astype(np.int16)
                linear_audios.append(linear_audio)

                diffusion_prediction = spectral_de_normalize_torch(diffusion_prediction).cpu().numpy()
                diffusion_audio = griffinlim(diffusion_prediction)[:min(diffusion_prediction.size(1), length) * self.hp.Sound.Frame_Shift]
                diffusion_audio = (diffusion_audio / np.abs(diffusion_audio).max() * 32767.5).astype(np.int16)
                diffusion_audios.append(diffusion_audio)
                
        files = []
        for index in range(diffusion_predictions.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (linear_prediction, diffusion_prediction, length, singer, lyric, linear_audio, diffusion_audio, file) in enumerate(zip(
            linear_predictions.cpu().numpy(),
            diffusion_predictions.cpu().numpy(),
            lengths.cpu().numpy(),
            singer_labels,
            lyrics,
            linear_audios,
            diffusion_audios,
            files
            )):
            title = 'Lyric: {}    Singer: {}'.format(lyric if len(lyric) < 90 else lyric[:90] + '…', singer)
            new_figure = plt.figure(figsize=(20, 5 * 4), dpi=100)
            ax = plt.subplot2grid((4, 1), (0, 0))
            plt.imshow(linear_prediction[:, :length], aspect='auto', origin='lower')
            plt.title('Linear Prediction    {}'.format(title))
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((4, 1), (1, 0))
            plt.plot(linear_audio[:length * self.hp.Sound.Frame_Shift])
            plt.title('Linear Audio    {}'.format(title))
            plt.margins(x= 0)            
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((4, 1), (2, 0))
            plt.imshow(diffusion_prediction[:, :length], aspect='auto', origin='lower')
            plt.title('Diffusion Prediction    {}'.format(title))
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((4, 1), (3, 0))
            plt.plot(diffusion_audio[:length * self.hp.Sound.Frame_Shift])
            plt.title('Diffusion Audio    {}'.format(title))
            plt.margins(x= 0)            
            plt.colorbar(ax= ax)
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)

            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.Linear.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                linear_audio
                )
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.Diffusion.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                diffusion_audio
                )
            
    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return
            
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_dict.values():
            model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, notes, durations, lengths, singers, genres, singer_labels, lyrics) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(tokens, notes, durations, lengths, singers, genres, singer_labels, lyrics, start_index= step * batch_size)

        for model in self.model_dict.values():
            model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model_dict['DiffSinger'].load_state_dict(state_dict['Model']['DiffSinger'])
        self.optimizer_dict['DiffSinger'].load_state_dict(state_dict['Optimizer']['DiffSinger'])
        self.scheduler_dict['DiffSinger'].load_state_dict(state_dict['Scheduler']['DiffSinger'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': {
                'DiffSinger': self.model_dict['DiffSinger'].state_dict(),
                },
            'Optimizer': {
                'DiffSinger': self.optimizer_dict['DiffSinger'].state_dict(),
                },
            'Scheduler': {
                'DiffSinger': self.scheduler_dict['DiffSinger'].state_dict(),
                },
            'Steps': self.steps
            }
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        torch.save(state_dict, checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)


    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model_dict = apply_gradient_allreduce(self.model_dict)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_path, hp_path)

        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)    
    argParser.add_argument('-p', '--port', default= 54321, type= int)
    argParser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl'
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()