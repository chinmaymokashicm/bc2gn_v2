from func_code.evaluate import Evaluate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


filepath_lr_model = "results/models/lr_model_2023-04-27_14-05-29.pkl"
filepath_siamese = "results/siameseBERT/siamese_bert_model.pth"
eval_obj = Evaluate(filepath_model_lr=filepath_lr_model, filepath_siamese=filepath_siamese, load_lr=True)

filepath_eval = "results/outputs/eval_test_2023-04-27_15-17-42.csv"
df = pd.read_csv(filepath_eval)

df = eval_obj.add_ground_truth(df)
df = eval_obj.remove_nulls(df)
df["siamese_pred"] = eval_obj.siamese_eval(df=df)

df.to_csv("test_results.csv", index=False)