    author: sam tenka
    credits: ian calloway, heming han, seth raker
    date: 2016-12-12
    descr: voice morphing system

# voice351 Project

## Creative Work

We had fun inventing many potential solutions to many problems
we encountered along the way. These include:
    
    Audio compression via Cepstral Transform & SVD (see `baseline/cepstral&inv_v1/`)
    Segmentation Metric (see `baseline/seg/`)
    Segmentation Algorithm via Pre-Convolution (see `baseline/segment/`)
    Dictionary-Conditioned Concatenation, with Cepstral Smoothing (see `baseline/encode/`)

We also implemented several techniques of relevance to this novel
application, including:

    Recurrent Phonetic Classifier (see `baseline/transcribe/`)
    Generative Adversarial Networks (see `gan/`)

Each of the above methods we tuned by hand or metric. For instance,
for the Cepstral smoothing, we binary searched coefficients for our
exponential moving average, validating by ear, until it became clear
that smoothing did not enhance our current results. Many other features
we tested and rejected in this way (another example is the use of Mel
Features rather than plain MFCC for our audio compression --- ultimately,
we use just Cepstra for this part of our project). So we designed our
global architecture iteratively.

More local choices we also tuned, often via a metric we designed. For
instance, our Segmenter has 4 parameters, and we optimized these (by
hand) to maximize our 'energy' metric. As another example, for each
neural phonetic classifier, we tuned (by hand) learning rate, the number
of cepstral coefficients used, and audio window size. 

Ultimately, we have several algorithms that outperform naive baselines
for each of the steps in our original Voice Morphing architecture. We
leave the task of composing together those pieces for future work.

## Demos

Our codebase is messy and fragmented. One challenge we encountered lied
in coordinating several members none of whom is a Computer Scientist
(our team is a wonderfully interdisciplinary mix: Linguistics, Electrical
Engineering, Computer Engineering, and Mathematics). Ultimately, we were
able to use git, to separate data such as hardcoded paths or large files
from code, and to begin linking our programs together. However, this work
remains unfinished. Many apologies for the catastrophe that is our software
organization.

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
