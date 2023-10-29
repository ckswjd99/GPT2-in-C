# GPT-in-C

GPT2 model running in C language.

It generates 64 tokens in 8.x seconds on Raspberry Pi 4, which is SOTA and lower bound of performance.

## Requirements

GPT-in-C uses [`openblas`](https://github.com/OpenMathLib/OpenBLAS) as a computing kernel.

```
$ sudo apt-get install libopenblas-dev
```

## Compile

```
$ make
```

## Get Weights

Pre-trained weights from the GPT2-124M model are required for inference.

`model/model_converter.py` automatically downloads the weights for PyTorch and then converts them to a format readable by GPT2-in-C. Alternatively, you can download the converted weights [here](https://huggingface.co/ckswjd99/GPT2-in-C/tree/main).

```
$ pip install torch
$ cd ./model
$ python ./model_converter.py

or

$ cd ./model
$ wget https://huggingface.co/ckswjd99/GPT2-in-C/resolve/main/GPT2-124M.mymodel
```

## Run

```
Usage: ./main.out [length] [batch_size]
```

Most options(context, temperature, beam search, etc.) are not available at this time. You can test the generation of text starting with some tokens(i.e. "Scientists"), or alternatively by replacing the start tokens in `gpt2.c`.

Here is an example with 64 tokens generated, which is identical to the output from the PyTorch version.

```
$ ./main.out 64 1
Loading GPT2 weights from ./model/GPT2-124M.mymodel
  Number of tensors: 196
  Finished loading weights!

Loading GPT2 tokenizer from ./vocabs.txt
  Finished loading tokneizer!

==================== OUTPUT TEXT ====================
Scientists, who have been studying the effects of the sun's radiation on the body, have found that the sun's rays are able to penetrate the skin and cause wrinkles and wrinkles.

The researchers found that the sun's rays can penetrate the skin and cause wrinkles and wrinkles.

"The sun's rays are able
===================================================

Inferenced with GPT2Model
Total ETA: 8070.774000 (ms)
```

