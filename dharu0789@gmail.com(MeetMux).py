
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


merged = pd.merge_asof(df_b.sort_values('ts'), df_a.sort_values('ts'), on='ts', direction='forward', suffixes=('_b','_a'))


dataset = merged[['text_a','text_b']].dropna().rename(columns={'text_a':'A_reply','text_b':'B_message'})

def build_example(prev_a_msgs, bmsg, reply, max_history_msgs=5):
    
    history = " ".join(prev_a_msgs[-max_history_msgs:])

    return {"context": f"History(A): {history}\nB: {bmsg}\nA:", "target": reply}



