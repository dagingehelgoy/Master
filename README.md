# An Continuous Approach to Text Generation using Generative Adversarial Networks

The project is the final delivery of the Master of Science (MSc) degree in Computer Science at the Department of Computer Science (IDI) at the Norwegian University of Science and Technology (NTNU).

## Abstract
Challenges with training GANs to produce discrete tokens has seen significant work in the past year. Generating text can be very useful for tasks such as dialog systems and image captioning. Applying deep generative models to text generation has seen limited success when compared to the visual domain and comes with the challenge of passing the gradient, while keeping the network differentiable. The known effective models that generate text with GAN extend the original framework proposed by Goodfellow using the REINFORCE loss function. We propose a novel approach that requires no modification to the training process proposed by Goodfellow and is able to produce meaningful text without any pre-training. One of the main problems with evaluating results produced by GANs is that there is no corresponding real data for each generated sample. To deal with this problem, we have developed an automatic evaluation method for text generative systems. This method combines the machine learning evaluation metric, BLEU, with a set of interchangeable information retrieval techniques. This allows us to evaluate the semantic quality of our models, as well as comparing them to a baseline.

## Results
The following are example sentences produced by our word embedding model.

### Flickr30k
`<sos>a newspaper is on a trees with the woods <eos> <pad>`<br />
`<sos>a man in his blue frisbee in the<eos> <pad> <pad>`<br />
`<sos>two men on a green on the bar <eos> <pad> <pad>`<br />
`<sos>a man of people are in the background log <eos> <pad>`<br />
`<sos>two men are holding on the room on the bar <eos>`<br />
`<sos>people are with a room and on the bar <eos> <pad>`<br />
`<sos>two men dancing in in the background <eos> <pad> <pad> <pad>`<br />
`<sos>people walking with leaps is hill <eos> <pad> <pad> <pad> <pad>`<br />
`<sos>a man wearing white are on the background <eos> <pad> <pad>`<br />
`<sos>a dog is playing at a <eos> <pad> <pad> <pad> <pad>`

### Oxford-102 Flower
`<sos>a flower with long and thick petals that are pink <eos>`<br />
`<sos>the petals are delicate and red pedals that are pink <eos>`<br />
`<sos>the petals on this flower are purple with red stamen <eos>`<br />
`<sos>this flower has petals that are pink with filament <eos>  <pad>`<br />
`<sos>this flower has petals that are pink with white stamen <eos>`<br />
`<sos>this flower has petals that are purple with red stamen <eos>`<br />
`<sos>this flower has petals that are purple with red stamen <eos>`<br />
`<sos>this flower has petals that are white and folded together <eos>`<br />
`<sos>this pink flower has flower flat and large white petals <eos>`<br />
`<sos>this pink flower has flower flat and large white petals <eos>`<br />

### Baseline Model
[SeqGAN](https://github.com/LantaoYu/SeqGAN)

## Usage
`python main.py --code gan`

## Requirements
h5py==2.6.0
Keras==1.2.2
scipy==0.18.1
scikit-learn==0.18.1
tensorflow-gpu==1.1.0rc2

