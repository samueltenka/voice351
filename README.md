    author: sam tenka
    credits: ian calloway, heming han, seth raker
    date: 2016-12-12
    descr: voice morphing system

# voice351 Project

## Creative Work

We had fun inventing many potential solutions to many problems
we encountered along the way. These include:
    
    Audio compression via MFCC & SVD, & Inverse MFCC (see `baseline/cepstral&inv_v1/`)
    Segmentation Metric (see `baseline/seg/`)
    Segmentation Algorithm via Pre-Convolution (see `baseline/segment/`)
    Dictionary-Conditioned Concatenation, with Cepstral Smoothing (see ``)

We also implemented several techniques of relevance to this novel
application, including:

    Recurrent Phonetic Classifier (see `baseline/transcribe`)
    Generative Adversarial Networks (see `gan/`)

## Demos

Our codebase is messy and fragmented. One challenge we encountered lied
in coordinating several members inexperienced with programming. Ultimately,
we were able to use git, to separate data such as hardcoded paths or large
files from code, and to begin linking our programs together. However,
this work remains unfinished. Many apologies for the catastrophe that is
our software organization.

Most of our work lies under `baseline/`.

### Phonetic Classifier

For instance, to train our phonetic classifier (accuracy ~30% on the very
challenging Buckeyes corpus, up from ~10% earlier today), one should:

    download the Buckeyes data (large; not recommended for demo)
    change the paths in `baseline/transcribe/Model_Trainer.py` to the location of that data 
    while run `python baseline/transcribe/Model_Trainer.py` complains about missing packages:
        install proper packages

Our experiments with cepstral embedding, segmentation scoring, and audio concatenation are similarly difficult to demo.

Two demos that, barring packages to be installed, are available immediately include: 

### Audio Segmentation via Convolutional Pre-Processing:

    cd into `baseline`
    run `python -m segment.convo`

A plot of segmented audio should appear.

### Adversarial Sampling of MNIST images 

    cd into `gan`
    view `gan_out.png` to see what output looks like after ~5 hours of training.
    run `python fetch_data.py` to download MNIST
    run `python train_gan.py` to train the GAN
        --> terminate when appropriate, e.g. ater 15 minutes. We auto-save the model every
            5 Generator/Discriminator alternations 
    run `python use_gan.py` to view output!
