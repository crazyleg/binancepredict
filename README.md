# Amateur-ish ML-powered crypto trading - true or false? 
this is very research code, don't expect it to work out-of-the-box, has bugs and requires some skill to operate. 

# Context
This is an exploratory/free-time project to undetstand if middle-age ML engineer is able to consistenly profit from trading using basic ML stack (a.k.a. alpha is within reach).

The idea if forked from Kaggle's G-research crypto prediction competition. By forked it means I am sticking to similar input data and doing prediction over 15m time horizon.

### Why 15 minutes?
 - G-Research likely knows something
 - \>15m (1h+) should be a land where fundamentals are much more important
 - < 15m (<1m) should be a land where HFT is playing the game and algos are more focused on that.
 - This is jsut another hyperparameter and hyperparameter space is vast in this domain. I am just randomly sticking to one that seems OKish.

### Current approach
The "dream" goal here is to build a core embedding/signal extraction system using deep NNs. I don't have domain knowledge/time to build it with traditional feature engineering. But I know convolutions real good, so my first sketches here revolve around assumption that convolutions will be capable of extracting signal similar to what techincal analyst does.

1. __Core ML__ Current solution is following:
   - Input: [B x [20 coins x 8 features = 180 features] x 512 timesteps] normalized over historical std. and mean.
   - Arch: Resnet1D + FC (no batchnorms)
   - Target: return  = (t<sub>n+15</sub>/t<sub>n</sub>)-1
   - Loss: L1

This is mostly by initial attemp to guess the initial parameters right. 

Loss: I tried L1 vs L2 and it seems L2 is significantly more unstanble.

Arch: On G-crypto dataset ResNet was giving less performance that hand-crafted TemporalConvolution, but here is seems to work.

I want this to __not require__ often retraining (like ones per day) and be robust enought to different market trends. This is likely naive cause even for training/evaluation one would want to have "expanding" window approach (seems typical for financial ML), but as a __think__ I am seening non-random result with existing approach I leave this of for now and put my stackes in the next step.


2. __ML post-processing <-> Risk management__

That's the thing I am currently iterating with is this step. On a validation part of the dataset one can obtain the [y, y_hat] tuple for each currency pair - question is how one determines the market entry condition based on that. That's a function of the risk/profit expectation. I am currently looking at some naive approaches, that are all based on a fact that one closes the position exactly after 15 minutes, no matter what happens. I advise agains going to market with that ;-)

Current: plot [threshold vs average profit@threshold], handpick threshold for each pair

Some other things: I also tried "normalizing" the output and making is more robust by adding and LR step on the [y, y_hat] dataset coming from validation. This seems to work, but is not implemented in this repo for now.

3. __Online__ prediction
   
Grabbing 512 timesteps from Binance API, running the network, and opening "fake" 15 minutes position at every opportuninty matching prediction result > threshold (or < then negative threshold for SELL operation) and logging everything. If good result obtained this would need to be connected to real Binance API.


## Stack and running the code
Uses ELK and neptune.ai for logging/monitoring - get your API keys, put to .env file.

1. Grabbing the training data.
   
   ```python dataprep/download_binance.py```

currently has some date hardcoded...find and fix them to most recent

2. Train the network

```python main.py```
Should output the best model and stats.pkl - normalization statistics for inference. 
Current "best" model and stats are stored in this repo under:
 - BSM4.py
 - stats.pkl

3. Generate threshold file following the structure in notebook: 

4. Start the ELK docker stack on a machine.
   1. [TODO] How-to
5. Run the prediction stack for online verification
    1. Take a look at kibana at localhost:5601

## Caveats

Data and -1 loss

## Help needed 
1. NN training and arch - requires lots of iterations to find best solution.
2. Generation of trading strategy from y_hat predictions.


# TODO 
## Before pushing to pub
 [ ] check for any api keys, etc
 [ ] add ml training code

## Fixes
 - [BUG] Prediction is rounded to 0 in ELK, so cannot visualize it  
 - requirements.txt

## Core ML - research
 - timestamp features
 - architectures
 - loss functions
 - activations

## Training
 - expanding window and finetuning