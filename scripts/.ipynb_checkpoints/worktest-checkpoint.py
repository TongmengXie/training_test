# As part of our research into CoT pathologies, we often train 'model organisms' of these pathologies. 
# These are LLMs which are specifically trained to have a particular pathology, which can be used to measure monitorability.
# This training can be done with supervised fine-tuning (SFT) and/or reinforcement learning (RL). 

# For this exercise, we will focus on SFT
# One pathology we can train in is semantic encoding. This is where a model produces text that 'looks' like realistic 
# English to an external monitor, but which obfuscates reasoning by substituting the semantics of this text.
# A straightforward example of this is simple word substitution, where specific words are replaced in the reasoning trace to
# evade monitors.

# Let's say that, for whatever reason, that we want to ban our model from using any digits in its reasoning, and we penalise
# it whenever it does so. If using numbers was important to the model to complete its task, then it might learn to substitute
# numbers with names of animals. Let's say:

# Zero = zebra
# One = owl
# Two = tiger
# Three = turtle
# Four = fox
# Five = ferret
# Six = snake
# Seven = shark
# Eight = eagle
# Nine = narwhal

# So that "42" becomes "fox tiger", and "9031" becomes "narwhal zebra turtle owl"

# To finetune our model organism, we can generate reasoning rollouts to maths answers, then perform the semantic substitution ourselves
# ourselves, then finetune the model to perform well with these reasoning traces.

# Let's run through the steps required to train our model organism.
# Please fill in the steps tagged with TODO. Feel free to choose your own model provider, data structures, etc.
# Please keep the TODO lines in, so that we can partition the steps of your script.

# NOTE: As this is an MVP, it is okay if your script does not cover every edge case. Instead, if you encounter edge cases but do not have time to remedy them you should make a note of them to discuss in the interview

# Before proceeding with each step, please describe the high-level logic of that part of the script, and the steps that you intend to go through

# You are free to use any tools available to you, but you will be expected to answer questions about your code in a technical interview afterwards

# +
# What could "edge cases" mean here?

# imports
# files handling
import pandas as pd
import json

# For rollouts generation
import os
from openai import OpenAI  # Inference Providers for deepseek-ai/DeepSeek-R1

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# -

# #### Stage 1 - data generation

# TODO: load in the arithmetic questions training set from train_questions.csv

# +
# Read provided files using pandas and json


train = pd.read_csv('../raw_data/train.csv')
test = pd.read_csv('../raw_data/test.csv')


with open("../raw_data/digits_to_animals.json", "r") as f:
    digits_to_animals = json.load(f)
    print(digits_to_animals)
# -

# TODO: generate reasoning traces for the training set

# +
# Should we ensure the same model for an "online" setting? - No, the model to train is too weak to generate reasoning traces
# Use Deepseek here:

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HG_API_KEY"),
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1:fireworks-ai",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)
# -

# TODO: substitute the digits for animal names
# Note that we are specifically encoding the reasoning. Thus, the answer produced by the model should not be encoded, and should be exactly the same as that provided by the model



# TODO: save the data

# #### Stage 2 - training

# TODO: load in the data you generated previously



# TODO: train on this data

# +
# the high-level logic: train the model. Rationale: Even if the reasoning is shallow, pathologies like
# 1.Does the monitor detect when digits are swapped for animals?
# 2.Does the model learn to always substitute numbers in traces but keep answers numeric?
# can still be studied
# (what should be the prompt, should it vary with every reasonign trace or invariable?) with CoT

# steps: 1. load model and tokeniser (set pad token)
#     2. hyper-parameters and wrap in configs
# 3. optional report (wandb/tensorboard) -> Did manual logging
#  4. train & eval loop
# 5. test on held-out


# -

# #### Stage 3 - evaluation and extension

# TODO: Run your trained model on the questions in test_questions.csv

# TODO: How well did the encoding train? This part is up to you!
# What metrics can you extract from these rollouts which would convincing whether or not your training worked?
# If your trained model does not do well on the test_question set, please write down additional training steps you would go through if you had more time to increase performance

additional training steps: 
    1.RLHF post-training 
    2. Aux loss term that penalises tokens corresponding to digits in earlier tokens (draw the threshold emperically or with sentivity test) in an output
    3. 

# TODO: Present the results of your in a way that you would be happy to present in a stand-up meeting
# Remember, we do NOT expect you to have a trained model, so we are happy if this looks more like an engineering update


