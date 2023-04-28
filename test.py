# from func_code.process import Process
# from func_code.metrics import Metrics
# from func_code.ml import Ml

# process_obj = Process()
# metrics_obj = Metrics()
# ml_obj = Ml()

# ml_obj.generate_datasets()

# from func_code.siameseBERT import SiameseBERT
# from func_code.evaluate import Evaluate
# import pandas as pd
# eval_obj = Evaluate(filepath_siamese="results/siameseBERT/siamese_bert_model.pth")
# model = SiameseBERT()
# np_siamese_pred = eval_obj.siamese_eval(train_or_test="test")
# df_test = eval_obj.df_test
# df_test["s_pred"] = pd.Series(np_siamese_pred)
# print(df_test.head())
# print(df_test.groupby("is_success")[["s_pred"]].agg(["count", "mean", "median"]))


from func_code.ml import Ml

ml_obj = Ml()

df = ml_obj.df_train.head(100)
ml_obj.create_feature_matrix(df=df)