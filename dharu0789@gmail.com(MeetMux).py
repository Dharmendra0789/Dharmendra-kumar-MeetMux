
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
import json, os

#Dataset Path
path_a = "/Desktop/Dataset/userA_chats.csv"
path_b = "/Desktop/Dataset/userB_chats.csv"


df_a = pd.read_csv(path_a)   
df_b = pd.read_csv(path_b)


df_a['ts'] = pd.to_datetime(df_a['timestamp'])
df_b['ts'] = pd.to_datetime(df_b['timestamp'])







