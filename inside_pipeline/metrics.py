# For EigenScore Helper
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import MinCovDet
from rouge_score import rouge_scorer
from sentence_transformers import util
import heapq
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore


###### 计算最后一个token特征的语义散度的作为句子的语义散度
###### 需要考虑每个句子的长度不一致，去除padding的token的影响
###### hidden_states : (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
def getEigenIndicator_v0(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    # selected_layer = -1
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        last_embeddings[ind,:] = hidden_states[num_tokens[ind]-2][selected_layer][ind,0,:]
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s


def get_num_tokens(generation):  # generation: num_seq x max(num_tokens)
    num_tokens = []
    for ids in generation:
        count = 0
        for id in ids:
            if id>2:
                count+=1
        num_tokens.append(count+1)
    return num_tokens
