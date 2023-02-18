# DiffSinger-KR
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Demo)](https://huggingface.co/spaces/codejin/DiffSingerKR)

This code is an implementation of DiffSinger for Korean. The algorithm is based on the following papers:
* [Liu, J., Li, C., Ren, Y., Chen, F., & Zhao, Z. (2022, June). Diffsinger: Singing voice synthesis via shallow diffusion mechanism. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 10, pp. 11020-11028).](https://arxiv.org/abs/2105.02446)
* [Xiao, Y., Wang, X., He, L., & Soong, F. K. (2022, May). Improving Fastspeech TTS with Efficient Self-Attention and Compact Feed-Forward Network. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 7472-7476). IEEE.](https://ieeexplore.ieee.org/document/9746408)


# Structure
* Structure is based on the DiffSinger, but I made some minor changes.
    * The multi-head attention is changed to linearized attention in FFT Block.
        * Positional encoding is removed.
    * Duration embedding is added.
        * It is based on the scaled positional encoding with very low initial scale.
    * Aux decoder and Diffusion are learned at the same time, not two stage.
* I changed several hyper parameters and data type
    * One of mel or spectrogram is can be selected as a feature type.
    * Token type is changed from phoneme to grapheme.* 

# Supported dataset

| Using  | | Dataset                                | Dataset Link                                                                              |
|--------|-|----------------------------------------|-------------------------------------------------------------------------------------------|
| O      | | Children's Song Dataset                | [Link](https://github.com/emotiontts/emotiontts_open_db/tree/master/Dataset/CSD)          |
| X      | | AIHub Korean Multi-Singer Song Dataset | [Link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=465) |

* I fixed some midi score to matching between note and wav F0.
* CSD dataset is used for the training of current checkpoint.
* [Pattern generator.py](./Pattern_Generator.py) supports the AIHub Dataset, but I did not used for the training of shared checkpoint.

# Hyper parameters
Before proceeding, please set the pattern, inference, and checkpoint paths in [Hyper_Parameters.yaml](Hyper_Parameters.yaml) according to your environment.

* Sound
    * Setting basic sound parameters.

* Tokens
    * The number of Lyric token.

* Notes
    * The highest note value for embedding.

* Duration
    * Min duration is used at pattern generating only.
    * Max duration is decided the maximum time step of model.
        MLP mixer always use the maximum time step.
    * Equality set the strategy about syllable to grapheme.
        * When `True`, onset, nucleus, and coda have same length or ±1 difference.
        * When `False`, onset and coda have Consonant_Duration length, and nucleus has duration - 2 * Consonant_Duration.

* Feature_Type
    * Setting the feature type (`Mel` or `Spectrogram`).

* Encoder
    * Setting the encoder(embedding).

* Diffusion
    * Setting the Diffusion denoiser.

* Train
    * Setting the parameters of training.

* Inference_Batch_Size
    * Setting the batch size when inference

* Inference_Path
    * Setting the inference path

* Checkpoint_Path
    * Setting the checkpoint path

* Log_Path
    * Setting the tensorboard log path

* Use_Mixed_Precision
    * Setting using mixed precision

* Use_Multi_GPU
    * Setting using multi gpu
    * By the nvcc problem, Only linux supports this option.
    * If this is `True`, device parameter is also multiple like '0,1,2,3'.
    * And you have to change the training command also: please check  [multi_gpu.sh](./multi_gpu.sh).

* Device
    * Setting which GPU devices are used in multi-GPU enviornment.
    * Or, if using only CPU, please set '-1'. (But, I don't recommend while training.)

# Generate pattern

```
python Pattern_Generate.py [parameters]
```
## Parameters
* -csd
    * The path of children's song dataset
* -am
    * The path of AIHub multi-singer song dataset
* -step
    * The note step that is explored when generating patterns.
    * The smaller step is, the more patterns are created in one song.
* -hp
    * The path of hyperparameter.
    
# Inference file path while training for verification.

* Inference_for_Training
    * There are three examples for inference.
    * It is midi file based script.

# Training

## Command

### Single GPU
```
python Train.py -hp <path> -s <int>
```

* `-hp <path>`
    * The hyper paramter file path
    * This is required.

* `-s <int>`
    * The resume step parameter.
    * Default is `0`.
    * If value is `0`, model try to search the latest checkpoint.

### Multi GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=32 python -m torch.distributed.launch --nproc_per_node=8 Train.py --hyper_parameters Hyper_Parameters.yaml --port 54322
```

* I recommend to check the [multi_gpu.sh](./multi_gpu.sh).

# Inference
* Please check [Inference.ipynb](./Inference.ipynb)

# Checkpoint
* Please check [Huggingface Space](https://huggingface.co/spaces/codejin/diffsingerkr/blob/main/Checkpoint/S_200000.pt)

# TODO
* Multi singer version version training with [AIHub Multi-Singer Song Dataset](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=465)