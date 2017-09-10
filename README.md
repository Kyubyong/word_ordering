# Word Ordering: Can Neural Networks Put a Scramble of Words in Correct Order?

When learning a second language, one of the hardest challenges is likely to be familiar with the word order. Word order can be also important in machine translation because translation is, roughly speaking, a process that one arranges words of target language which are equivalent to source language in order. Probably you've done a word scramble game where you are to put shuffled words or letters in the original order. I think it's quite fun to see if neural networks can do it. Okay. Can you order the following words correctly?

**can translation machine order also important be word in**

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow == 1.2 (Probably 1.3 should work, too, though I didn't test it)
  * matplotlib
  * distance
  * tqdm

## Model Architecture
I employ the Transformer which was introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). It's known to the state-of-the-art model in the machine translation task as of 2017. However, I don't know if it fits in this task the best. Actually, I think a simpler architecture may work. The figure below is borrowed from the paper.

<img src="fig/transformer.png">

## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `data_load.py` contains functions regarding loading and batching data.
  * `modules.py` has all building blocks for encoder/decoder networks.
  * `train.py` contains the model and training code.
  * `eval.py` is for evaluation and inference.

## Training
* STEP 1. Download and extract [Leipzig English News 2015 1M Corpus](http://wortschatz.uni-leipzig.de/en/download).
* STEP 2. Adjust hyper parameters in `hyperparams.py` if necessary.
* STEP 3. Run `train.py` or download the [pretrained files](https://www.dropbox.com/s/5axxz6f9g93ms72/logdir.zip?dl=0).

## Training Loss and Accuracy
* Training Loss
<img src="fig/mean_loss.png">

* Training Accuracy
<img src="fig/acc.png">

## Evaluation
  * Run `eval.py`. 

We take WER (Word Error Rate) as the metric. WER is computed as follows:

WER = Edit distance / Number of words

**Total WER : 10731/23541=0.46**

Check the `results` folder for details.
