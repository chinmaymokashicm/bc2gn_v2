"""
Generate evaluation results for classic ML and Siamese network
"""
import pandas as pd
import os, joblib, torch

from .ml import Ml
from .siameseBERT import SiameseBERTDataset, SiameseBERT
from .process import Process
from tqdm import tqdm

class Evaluate:
    def __init__(self, filepath_model_lr=None, filepath_siamese=None, load_lr=False, load_siamese=False):
        tqdm.pandas()
        self.folderpath_datasets = "results/datasets"
        self.folderpath_siamese = "results/siameseBERT"
        self.folderpath_models = "results/models"
        self.process_obj = Process()
        self.df_test = pd.read_csv(os.path.join(self.folderpath_datasets, "test.csv"))
        self.df_train = pd.read_csv(os.path.join(self.folderpath_datasets, "train.csv"))
        self.df = {
            "train": self.df_train,
            "test": self.df_test
        }
        if load_lr:
            if filepath_model_lr is None:
                filepath_model_lr = [os.path.join(self.folderpath_models, filename) for filename in os.listdir(self.folderpath_models) if filename.endswith("pkl")][0]
            self.model_lr = joblib.load(filepath_model_lr)
        if filepath_siamese is None:
            filepath_siamese = "results/siameseBERT/best_model.pth"
        if load_siamese:
            self.model_siamese = SiameseBERT()
            self.model_siamese.load_state_dict(torch.load(filepath_siamese))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_siamese.to(device)
            self.model_siamese.eval()
            self.ml_obj = Ml()
        
    def load_lr_eval2df(self, filepath_eval, train_or_test="test"):
        return self.ml_obj.evaluate(df=self.df[train_or_test], filepath_eval=filepath_eval, model=self.model_lr, save=False)
    
    def add_ground_truth(self, df):
        df["is_success"] = df.apply(lambda row: 1 if row["entrez_id_text"] == row["entrez_id_term"] else 0, axis=1)
        return df
    
    def remove_nulls(self, df):
        df = df[~(df["text"].isnull() | df["terms"].isnull())]
        return df
    
    def siamese_eval(self, train_or_test="test", df=None):
        if df is None:
            df = self.df[train_or_test]
        dataloader = SiameseBERTDataset(text1=df["text"].to_numpy(), text2=df["terms"].to_numpy(), label=df["is_success"].to_numpy())
        return self.model_siamese.predict_proba(dataloader)
        
    def load_all_eval(self, filepath_eval, train_or_test="test"):
        df = self.load_lr_eval2df(filepath_eval=filepath_eval, train_or_test=train_or_test)
        # df = self.df[train_or_test]
        df["siamese_pred"] = pd.Series(self.siamese_eval(train_or_test=train_or_test))
        df["siamese_pred"] = df["siamese_pred"].apply(float)
        self.df_eval = df
        return df
    
    def filter_successful(self, df):
        id_counts = df.groupby("entrez_id_term")["is_success"].agg(["sum", "count"]).reset_index()
        list_id_success = id_counts[id_counts["sum"] > 0]["entrez_id_term"].tolist()
        df_filtered = df[df["entrez_id_term"].isin(list_id_success)]
        return df_filtered
    
    def entrez_id_rankings(self, df, pred_colname="pred", rank_colname="rank", top_N=10):
        # if df is None:
        #     df = self.load_all_eval(train_or_test=train_or_test)
        # return df.groupby("entrez_id_term").apply(lambda x: x.nlargest(top_N, pred_colname)).reset_index(drop=True)
        df[rank_colname] = df.groupby("entrez_id_term")[pred_colname].rank(method="first", ascending=False)
        df[rank_colname] = df[rank_colname].astype(int)
        return df[df[rank_colname] <= top_N][["entrez_id_term", "text", pred_colname, rank_colname, "is_success"]]

    def get_successful_match_stats(self, df):
        id_counts = df.groupby("entrez_id_term")["is_success"].agg(["sum", "count"])
        id_with_0_only_count = id_counts[id_counts["sum"] == 0]["count"].count()
        id_with_1_count = id_counts[id_counts["sum"] > 0]["count"].count()
        return {
            "fail": id_with_0_only_count, 
            "success": id_with_1_count
        }
        
    def get_successful_per_rank(self, df, pred_colname="pred", rank_colname="rank", top_N=10):
        df_filtered = self.filter_successful(df)
        df_ranked = self.entrez_id_rankings(df_filtered, pred_colname=pred_colname, rank_colname=rank_colname, top_N=top_N)
        return self.get_successful_match_stats(df_ranked)
    
    def combine_test_with_lexicon(self, df_test=None, save_to=None):
        dict_lexicon = {
            dict_pair["entrez_id"]: dict_pair["terms"]
                for dict_pair in self.process_obj.read_lexicon()
        }
        if df_test is None:
            df_test = self.df_test
        list_unique_term_ids = df_test["entrez_id_term"].unique().tolist()
        dict_lexicon = {entrez_id: list_terms for entrez_id, list_terms in dict_lexicon.items() if int(entrez_id) in list_unique_term_ids}
        list_dict_rows = []
        for entrez_id, list_terms in dict_lexicon.items():
            for term in list_terms:
                list_dict_rows.append({
                    "entrez_id_lexicon": entrez_id,
                    "gene_lexicon": term
                })
        df_lexicon = pd.DataFrame(list_dict_rows)
        df_test = df_test[["entrez_id_text", "text"]]
        df = df_lexicon.merge(right=df_test, how="cross")
        df["is_success"] = df.apply(lambda row: 1 if row["entrez_id_text"] == row["entrez_id_lexicon"] else 0, axis=1)
        if save_to is not None:
            df.to_csv(save_to, index=False)
        return df
    
    
    def generate_multi_eval_results(self, df=None, train_or_test="test", top_N_ranks=[2, 5, 10, 20, 30, 40, 50, 100, 200]):
        dict_results = {"lr": {}, "siamese": {}}
        if df is None:
            df = self.df[train_or_test]
            
        df_filtered = self.filter_successful(df)
        list_dict_rows = []
        
        for lr_or_siamese, pred_colname, rank_colname in [("lr", "pred", "lr_rank"), ("siamese", "siamese_pred", "siamese_rank")]:
            for rank in top_N_ranks:
                df_ranked = self.entrez_id_rankings(df_filtered, pred_colname=pred_colname, rank_colname=rank_colname, top_N=rank)
                dict_results = self.get_successful_match_stats(df_ranked)
                success, fail = dict_results["success"], dict_results["fail"]
                dict_row = {
                    "model": lr_or_siamese,
                    "top": rank,
                    "success": success,
                    "fail": fail,
                    "accuracy_perc": success * 100 / (success + fail)
                }
                list_dict_rows.append(dict_row)
        
        return pd.DataFrame(list_dict_rows)
                