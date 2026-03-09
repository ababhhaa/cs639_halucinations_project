#Generating k reponses for the same prompt, getting hidden states, computing embeddings
import argparse
import glob
import json
import os
import copy
import time

import pandas as pd
import torch
import tqdm
import transformers
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore

import _settings
import dataeval.coqa as coqa
import dataeval.nq_open as nq_open
import dataeval.triviaqa as triviaqa
import dataeval.SQuAD as SQuAD
import dataeval.TruthfulQA as TruthfulQA
import models
import utils
from func.metric import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)


args = parser.parse_args()
logInfo = open("./data/output/logInfo_{}_{}.txt".format(args.model, args.dataset), mode="w",encoding="utf-8")


dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                            num_beams=1, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                            temperature=args.temperature, generation_config=generation_config,
                            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                            )
