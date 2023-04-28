"""
Code related to machine learning related processes
"""
import os, csv, bisect, json, torch, ast
from itertools import islice
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
tqdm().pandas()
# try:
#     from tqdm.notebook import tqdm
# except ImportError:
#     from tqdm import tqdm
import random
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .process import Process
from .metrics import Metrics

class Ml:
    def __init__(self):
        random.seed(2023)
        self.folderpath_pairs = "results/filtered_pairs"
        self.folderpath_dataset = "results/datasets"
        self.process_obj = Process()
        self.metrics_obj = Metrics()
        self.folderpath_model = "/data/cmokashi/bc2gm/named-entity-recognition/output/BC2GM"
        self.tokenizer = BertTokenizer.from_pretrained(self.folderpath_model)
        self.bert_model = BertModel.from_pretrained(self.folderpath_model)
        self.list_colnames = ["abstract_id", "entrez_id_text", "text", "terms", "entrez_id_term","is_acronym", "common_tokens", "bigram_similarity", "is_small_string_substring", "is_pass_filter", "prefix_combined","suffix_combined", "is_same_numbers", "different_tokens","soft_tfidf_similarity", "levenshtein_similarity","jaro_winkler_similarity"]
        self.list_colnames2edit = ["common_tokens", "prefix_combined","suffix_combined", "is_same_numbers", "different_tokens"]
        self.list_colnames4embeddings = ["common_tokens", "prefix_combined", "suffix_combined", "different_tokens"]
        self.list_input_colnames = ["common_tokens", "bigram_similarity","is_small_string_substring", "is_pass_filter", "prefix_combined","suffix_combined", "is_same_numbers", "different_tokens", "soft_tfidf_similarity", "levenshtein_similarity","jaro_winkler_similarity"]
        self.df_train = self.tweak_dtypes(df=pd.read_csv(os.path.join(self.folderpath_dataset, "train.csv")))
        self.df_test = self.tweak_dtypes(df=pd.read_csv(os.path.join(self.folderpath_dataset, "test.csv")))
                
    def prepare_model(self):
        # Load training data
        df_train = self.df_train
        # Create feature matrix (including output) of training data
        print("Creating training feature matrix")
        list_train_input_features, list_train_output = self.create_feature_matrix(df=df_train)
        # Train model
        print("Features generated. Training model now")
        self.train_LR(list_train_input_features, list_train_output)
        # Save model
        print("Model trained. Saving model")
        self.save_model(self.lr_model)
        print("Model saved")
        return self.lr_model
        
    def generate_evaluation_results(self, df_train=None, df_test=None, model=None):
        df_eval_train = self.evaluate(df=df_train, filename_result_prefix="eval", train_or_test="train", model=model)
        print("Generated evaluation results for training data")
        df_eval_test = self.evaluate(df=df_test, filename_result_prefix="eval", train_or_test="test", model=model)
        print("Generated evaluation results for test data")
        return df_eval_train, df_eval_test
    
    def get_syn_split_stats(self):
        dict_n_rows = {}
        folderpath_data = self.folderpath_dataset
        for filename in os.listdir(folderpath_data):
            if filename.startswith("syn_"):
                with open(os.path.join(folderpath_data, filename), "r") as f:
                    is_syn = filename.split("_")[1]
                    if is_syn not in dict_n_rows:
                        dict_n_rows[is_syn] = {}
                    dict_n_rows[is_syn][filename.split("_")[2].split(".")[0]] = sum(1 for _ in f)
        return dict_n_rows
    
    def generate_features(self, list_text):
        # Tokenize all the strings in the list and concatenate the resulting tokens
        concatenated_tokens = []
        for text in list_text:
            text = str(text)
            tokens = self.tokenizer.tokenize(text)
            concatenated_tokens.extend(tokens)
        
        concatenated_text = " ".join(concatenated_tokens)
        
        # Set a maximum length for the concatenated text and truncate or pad it as needed
        max_length = 512
        if len(concatenated_text) > max_length:
            concatenated_text = concatenated_text[:max_length]
        else:
            num_padding = max_length - len(concatenated_text)
            padding_text = "[PAD]" * num_padding
            concatenated_text += padding_text
        
        # Convert the padded/truncated text to input features
        encoded_text = self.tokenizer.encode_plus(
            concatenated_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Generate the word embeddings and take the average
        self.bert_model.to("cuda")
        encoded_text = encoded_text.to("cuda")
        with torch.no_grad():
            model_output = self.bert_model(**encoded_text)
            embeddings = model_output.last_hidden_state
            avg_embeddings = torch.mean(embeddings, axis=1)
                
        
        return avg_embeddings.cpu().numpy()
                
    def generate_datasets(self, class_ratio=8):
        # Pick all synonymous pairs
        # Pick (N times of synonymous) non-synonymous pairs
        dict_n_rows = self.get_syn_split_stats()
        for train_or_test in ["train", "test"]:
            n_rows_syn_1 = dict_n_rows["1"][train_or_test]
            n_rows_syn_0 = dict_n_rows["0"][train_or_test]
            n_rows_syn_0_allowed = int(class_ratio * n_rows_syn_1)
            print(f"Number of allowed rows for non-syn in {train_or_test}: {n_rows_syn_0_allowed}/{n_rows_syn_0}")
            list_row_ids = random.sample(range(0, n_rows_syn_0), n_rows_syn_0_allowed)
            filepath = os.path.join(self.folderpath_dataset, f"syn_0_{train_or_test}.csv")
            list_dict_rows = self.process_obj.read_csv_rows_by_ids(
                filepath_csv=filepath,
                list_row_ids=list_row_ids,
                is_dict=True,
                list_colnames=["index"] + self.list_colnames
            )
            print(f"Number of rows in {train_or_test}: {len(list_dict_rows)}")
            # list_dict_rows = [{key: value for key, value in dict_row if key != "index"} for dict_row in list_dict_rows]
            df = pd.DataFrame(list_dict_rows)
            df = df.drop(columns=["index"])
            print(f"Number of rows in {train_or_test} df: {len(df.index)}")
            df_syn_1 = pd.read_csv(os.path.join(self.folderpath_dataset, f"syn_1_{train_or_test}.csv"), names=["index"] + self.list_colnames)
            df_syn_1 = df_syn_1.drop(columns=["index"])
            print(f"Number of rows in {train_or_test} df_syn_1: {len(df_syn_1.index)}")
            df_combined = pd.concat([df, df_syn_1], ignore_index=True)
            df_combined.to_csv(os.path.join(self.folderpath_dataset, f"{train_or_test}.csv"))
            print("\n\n\n")
                
    def tweak_dtypes(self, list_colnames2edit=None, df=None, train_or_test="train"):
        if list_colnames2edit is None:
            list_colnames2edit = self.list_colnames2edit
        # for train_or_test in ["train", "test"]:
        if df is None:
            filepath = os.path.join(self.folderpath_dataset, f"{train_or_test}.csv")
            df = pd.read_csv(filepath)
        for colname in list_colnames2edit:
            df[colname] = df[colname].apply(lambda item: ast.literal_eval(str(item)))
        df = df[self.list_colnames]
        return df
            
    def create_feature_matrix(self, train_or_test="train", df=None):
        if df is None:
            # filepath = os.path.join(self.folderpath_dataset, f"{train_or_test}.csv")
            # df = pd.read_csv(filepath)
            if train_or_test == "train":
                df = self.df_train
            else:
                df = self.df_test
        list_features = []
        is_success = df.apply(lambda row: 1 if row["entrez_id_text"] == row["entrez_id_term"] else 0, axis=1).tolist()
        filepath_features_train_or_test = os.path.join(self.folderpath_dataset, f"{train_or_test}.npy")
        if os.path.exists(filepath_features_train_or_test):
            return np.load(filepath_features_train_or_test, allow_pickle=True), is_success
        for colname in tqdm(self.list_input_colnames):
            if colname in self.list_colnames4embeddings:
                filepath_train_or_test_colname = os.path.join(self.folderpath_dataset, f"{train_or_test}_{colname}.npy")
                if os.path.exists(filepath_train_or_test_colname):
                    features_colname = np.load(filepath_train_or_test_colname, allow_pickle=True)
                else:
                    print(f"Generating features for {colname}")
                    features_colname = np.vstack(df[colname].progress_apply(self.generate_features))
                    list_features.append(features_colname)
                    features_colname = np.asarray(features_colname)
                    np.save(filepath_train_or_test_colname, features_colname)
            else:
                list_features.append(df[colname].values.reshape(-1, 1))
        # np_input_features = np.concatenate(list_features)
        np_input_features = np.concatenate(list_features, axis=1)
        np.save(filepath_features_train_or_test, np_input_features)
        return np_input_features, is_success
    
    def impute_and_scale(self, X):
        # Impute inf values with max value
        X[X == np.inf] = np.max(X[X != np.inf])

        imputer = SimpleImputer(strategy='mean', fill_value='NaN')
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        return X_scaled
    
    def train_LR(self, feature_matrix, list_output):
        feature_matrix = self.impute_and_scale(feature_matrix)
        self.lr_model = LogisticRegression(random_state=0, max_iter=1000, verbose=1)
        self.lr_model.fit(feature_matrix, list_output)
        
    def save_model(self, model, prefix="lr_model"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{prefix}_{timestamp}.pkl"
        filepath = os.path.join(self.process_obj.foldername_results, self.process_obj.dict_foldernames["models"], filename)
        joblib.dump(model, filepath)
        
    def load_model(self, filename, overwrite=False):
        filepath = os.path.join(self.process_obj.foldername_results, self.process_obj.dict_foldernames["models"], filename)
        if overwrite:
            self.lr_model = joblib.load(filepath)
            return "Success"
        else:
            return joblib.load(filepath)
        
    # def predict_LR(self, feature_matrix, list_output):
    #     if self.lr_model is None:
    #         raise Exception("Train the model first")
    #     list_predicted_labels = self.lr_model.predict(feature_matrix)
    #     return {
    #         "accuracy": accuracy_score(list_output, list_predicted_labels),
    #         "precision": precision_score(list_output, list_predicted_labels),
    #         "recall": recall_score(list_output, list_predicted_labels),
    #         "f1": f1_score(list_output, list_predicted_labels),
    #         "roc_auc": roc_auc_score(list_output, list_predicted_labels)
    #     }
    
    def predict_proba_LR(self, feature_matrix, model=None):
        if model is None:
            model = self.lr_model
        if model is None:
            raise Exception("Train the model first")
        return model.predict_proba(feature_matrix)[:, 1]
    
    # def generate_metrics(self, string_a, string_b):
    #     dict_input = {
    #         "text": string_a,
    #         "terms": string_b,
    #         "is_acronym": 1 if self.metrics_obj.is_small_string_acronym_long_string(string_a, string_b) else 0,
    #         "common_tokens": self.metrics_obj.get_common_tokens(string_a, string_b),
    #         "bigram_similarity": self.metrics_obj.get_ngram_similarity(self.metrics_obj.get_character_ngrams(string_a), self.metrics_obj.get_character_ngrams(string_b)),
    #         "is_small_string_substring": 1 if self.metrics_obj.is_small_string_substring(string_a, string_b) else 0,
    #         "prefix_combined": self.metrics_obj.get_prefixes(string_a, string_b),
    #         "suffix_combined": self.metrics_obj.get_suffixes(string_a, string_b),
    #         "is_same_numbers": 1 if self.metrics_obj.is_same_numbers(string_a, string_b) else 0,
    #         "different_tokens": self.metrics_obj.get_different_tokens(string_a, string_b),
    #         "soft_tfidf_similarity": self.metrics_obj.soft_tfidf_similarity(string_a, string_b),
    #         "levenshtein_similarity": self.metrics_obj.levenshtein_similarity(string_a, string_b),
    #         "jaro_winkler_similarity": self.metrics_obj.jaro_winkler_similarity(string_a, string_b)
    #     }
    #     dict_input["is_pass_filter"] = 1 if (dict_input["bigram_similarity"] >= 0.5 and dict_input["is_small_string_substring"]) else 0
    
    # def generate_similarity_score(self, string_a, string_b, model=None):
    #     if model is None:
    #         model = self.lr_model
    #     if model is None:
    #         raise Exception("Train the model first")
    
    def evaluate(self, filepath_eval=None, df=None, filename_result_prefix="eval", model=None, save=True, train_or_test="train"):
        if filepath_eval is not None:
            return pd.read_csv(filepath_eval)
        if df is None:
            # df = self.tweak_dtypes(train_or_test=train_or_test)
            if train_or_test == "train":
                df = self.df_train
            else:
                df = self.df_test
        filepath_npy = os.path.join(self.folderpath_dataset, f"{train_or_test}.npy")
        if os.path.exists(filepath_npy):
            list_input_features = np.load(filepath_npy, allow_pickle=True)
            print(f"Found {filepath_npy}!!")
        else:
            # Create feature matrix from df
            list_input_features, _ = self.create_feature_matrix(df=df, train_or_test=train_or_test)
        list_input_features = self.impute_and_scale(list_input_features)
        # Use model to get prediction
        if model is None:
            model = self.lr_model
        if model is None:
            filename_model = [filename for filename in os.listdir(os.path.join(self.process_obj.foldername_results, self.process_obj.dict_foldernames["models"]))][0]
            self.load_model(filename_model, overwrite=True)
            model = self.lr_model
        # Append result to df and return
        df["pred"] = self.predict_proba_LR(feature_matrix=list_input_features, model=model)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if save:
            df.to_csv(os.path.join(self.process_obj.foldername_results, self.process_obj.dict_foldernames["outputs"], f"{filename_result_prefix}_{train_or_test}_{timestamp}.csv"))
        return df
    
if __name__ == "__main__":
    ml_obj = Ml()
    print(ml_obj.shrink_dataset_by_name())