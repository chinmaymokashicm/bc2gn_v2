"""
Code related to processing data
"""
import logging, time, multiprocessing, os, json, csv
from itertools import islice
import pandas as pd
# from tqdm.auto import tqdm
from tqdm import tqdm
# tqdm.pandas()
from datetime import datetime
from .metrics import Metrics

class Process:
    def __init__(self, log=False, filepath_log=None, logger_class="Generate pairs"):
        self.filepath_lexicon = "original_data/biocreative2normalization/entrezGeneLexicon.list"
        self.filepath_train = "original_data/biocreative2normalization/trainingData/training.genelist"
        self.filepath_test = "original_data/bc2GNtest/bc2GNtest.genelist"
        self.metrics_obj = Metrics()
        self.list_colnames = ["abstract_id", "entrez_id_text", "text", "terms", "entrez_id_term"]
        
        if filepath_log is None:
            time_now = str(time.asctime(time.localtime())).replace(" ", "_")
            filepath_log = f"{time_now}.log"
        if log:
            print(f"Logging to {filepath_log}")
            logging.basicConfig(filename=filepath_log,
                    filemode="a",
                    format="%(asctime)s,%(msecs)d----%(name)s----%(levelname)s----%(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.DEBUG)
        else:
            logging.disable(logging.CRITICAL)
        self.logger = logging.getLogger(logger_class)
        
        self.dict_foldernames = {
            "term_text_pairs": "term_text_pairs",
            "ngrams": "ngrams",
            "aggregations": "aggregations",
            "datasets": "datasets",
            "stats": "stats",
            "filtered_pairs": "filtered_pairs",
            "models": "models",
            "outputs": "outputs"
        }
        
        # self.filename_term_text_pairs = "term_text_pairs.csv"
        self.foldername_results = "results"

        for foldername in self.dict_foldernames.values():
            os.makedirs(os.path.join(self.foldername_results, foldername), exist_ok=True)
    
    def get_stats(self):
        n_terms = sum([len(dict_pair["terms"]) for dict_pair in self.read_lexicon()])
        n_text_train = sum([len(dict_pair["text"]) for dict_pair in self.read_data(train_or_test="train")])
        n_text_test = sum([len(dict_pair["text"]) for dict_pair in self.read_data(train_or_test="test")])
        
        return n_terms, n_text_train, n_text_test
    
    def load_filtered_filepaths(self, train_or_test="train"):
        list_filepaths = [os.path.join(os.path.join(self.foldername_results, "filtered_pairs"), filename) for filename in os.listdir(os.path.join(self.foldername_results, "filtered_pairs")) if train_or_test in filename]
        return list_filepaths
    
    def chunkify(self, list_items, chunksize=100):
        for i in range(0, len(list_items), chunksize):
            yield list_items[i: i + chunksize]
        
    # def read_csv_rows_by_ids(self, filepath_csv, list_row_ids, chunk_size=1000, is_dict=False, list_colnames=None):
    #     list_selected_rows = []
    #     with open(filepath_csv, 'r') as f:
    #         if is_dict:
    #             reader = csv.DictReader(f, fieldnames=list_colnames)
    #         else:
    #             reader = csv.reader(f)
    #         list_row_ids = sorted(list_row_ids)
    #         pbar = tqdm(total=chunk_size)
    #         while list_row_ids:
    #             list_rows = []
    #             min_id, max_id = list_row_ids[0], list_row_ids[0] + chunk_size - 1
    #             list_selected_ids = [id for id in list_row_ids if min_id <= id <= max_id]
    #             for row_id, row in enumerate(reader):
    #                 if row_id in list_selected_ids:
    #                     list_rows.append(row)
    #                     list_selected_ids.remove(row_id)
    #                 if not list_selected_ids:
    #                     break
    #             list_selected_rows.extend(list_rows)
    #             pbar.update(1)
    #             list_row_ids = [id for id in list_row_ids if id not in list_selected_ids]
    #     return list_selected_rows
    
    def read_csv_rows_by_ids(self, filepath_csv, list_row_ids, is_dict=False, list_colnames=None):
        list_selected_rows = []
        with open(filepath_csv, 'r') as f:
            if is_dict:
                reader = csv.DictReader(f, fieldnames=list_colnames)
            else:
                reader = csv.reader(f)
            list_row_ids = sorted(list_row_ids)
            pbar = tqdm(total=len(list_row_ids))
            for row_id, row in enumerate(reader):
                pbar.update(1)
                if row_id == list_row_ids[0]:
                    list_selected_rows.append(row)
                    list_row_ids.pop(0)
                    if len(list_row_ids) == 0:
                        break
                else:
                    continue
        return list_selected_rows
    
    def read_lexicon(self, filepath=None):
        """Loads lexicon data

        Args:
            filepath (str, optional): Filepath. Defaults to None.

        Returns:
            list: List of rows
        """
        if filepath is None:
            filepath = self.filepath_lexicon
        list_pairs = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                entrez_id = line.split("\t")[0]
                list_terms = [f'"{gene.strip()}"' for gene in line.split("\t")[1:]] # Adding quotes to every gene since it may contain commas
                list_pairs.append({
                    "entrez_id": entrez_id,
                    "terms": list_terms
                })
        return list_pairs
    
    def read_data(self, train_or_test="train"):
        """Reads train or test data

        Returns:
            list: List of rows
        """
        if train_or_test == "train":
            filepath = self.filepath_train
        elif train_or_test == "test":
            filepath = self.filepath_test
        else:
            raise Exception("Choose 'train' or 'test'")
        list_rows = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                abstract_id = line.split("\t")[0]
                entrez_id = line.split("\t")[1]
                list_annotations = [f'"{line.strip()}"' for line in line.split("\t")[2:]]
                list_rows.append({
                    "abstract_id": abstract_id, 
                    "entrez_id": entrez_id, 
                    "text": list_annotations
                    })
        return list_rows
    
    def write_to_file(self, filepath, row, flush=False):
        """Writes row to file

        Args:
            filepath (str): Filepath
        """
        with open(filepath, "a") as f:
            len_row = len(row)
            str_row = ""
            sep_counter = 0
            for item in row:
                if type(item) == str:
                    str_row += f'"{item}"'
                else:
                    str_row += str(item)
                sep_counter += 1
                if sep_counter < len_row:
                    str_row += ","
            f.write(str_row)
            f.write("\n")
            if flush:
                f.flush()
    def remove_quotes(self, term, quote='"'):
        """Replaces quotations from the string

        Args:
            term (str): String
            quote (str, optional): Quotation. Defaults to '"'.

        Returns:
            str: Modified string
        """
        return term.replace(quote, "")
                
    def write_multiple_rows_to_file(self, list_rows, filepath):
        for row in list_rows:
            self.write_to_file(filepath, row)
    
    def generate_term_text_pairs(self, n_rows_per_group=1000):
        """Generates term-text pairs. Distributes the pairs across several files

        Args:
            n_rows_per_group (int, optional): Number of rows per group. Defaults to 1000.
        """
        df_lexicon = pd.DataFrame(self.read_lexicon())
        for train_or_test in tqdm(["train", "test"], desc="Dataset"):
            self.write_to_file(filepath=os.path.join(self.foldername_results, self.dict_foldernames["term_text_pairs"], f"term_text_pairs_{train_or_test}_colnames.csv"), row=self.list_colnames)
            df = pd.DataFrame(self.read_data(train_or_test=train_or_test))
            df_merged = pd.merge(left=df_lexicon, right=df, how="cross", suffixes=["_text", "_term"])
            list_df_rows = df_merged.to_dict("records")
            list_df_rows_2D = [list_df_rows[i:i+n_rows_per_group] for i in range(0, len(list_df_rows), n_rows_per_group)]
            for iteration, list_dict_rows in tqdm(enumerate(list_df_rows_2D), desc="Iterations"):
                filepath = os.path.join(self.foldername_results, self.dict_foldernames["term_text_pairs"], f"term_text_pairs_{train_or_test}_{iteration}.csv")
                list_rows = []
                for dict_row in list_dict_rows:
                    # Insert columns in correct order
                    for text in dict_row["text"]:
                        for term in dict_row["terms"]:
                            row = []
                            for colname in self.list_colnames:
                                if colname == "text":
                                    row.append(self.remove_quotes(text))
                                elif colname == "terms":
                                    row.append(self.remove_quotes(term))
                                else:
                                    row.append(dict_row[colname])
                            list_rows.append(row)
                    
                p = multiprocessing.Process(target=self.write_multiple_rows_to_file, args=[list_rows, filepath])
                p.start()
                if iteration % 50:
                    p.join()

    def load_term_text_pairs_filepaths(self):
        dict_results = {}
        for train_or_test in ["train", "test"]:
            # results/term_text_pairs/term_text_pairs_train_colnames.csv
            list_filepaths = [os.path.join(
                self.foldername_results, 
                self.dict_foldernames["term_text_pairs"],
                filename
                ) for filename in os.listdir(os.path.join(
                    self.foldername_results, 
                    self.dict_foldernames["term_text_pairs"]))
                if filename.startswith(f"term_text_pairs_{train_or_test}")
                and not filename.endswith("colnames.csv")]
            dict_results[train_or_test] = list_filepaths
        return dict_results

    def get_term_text_filter_stats(self):
        dict_filepaths = self.load_term_text_pairs_filepaths()
        dict_results = {}
        for train_or_test in ["train", "test"]:
            list_filepaths = dict_filepaths[train_or_test]
            n_pass, n_fail, n_total = 0, 0, 0
            for filepath in tqdm(list_filepaths):
                df = pd.read_csv(filepath, names=self.list_colnames)
                df["bigram_similarity"] = df.apply(lambda row: self.metrics_obj.get_ngram_similarity(self.metrics_obj.get_character_ngrams(row["text"]), self.metrics_obj.get_character_ngrams(row["terms"])), axis=1)
                df["is_small_string_substring"] = df.apply(lambda row: self.metrics_obj.is_small_string_substring(row["terms"], row["text"]), axis=1)
                n_pass_current = len(df[(df["bigram_similarity"] >= 0.5) & (df["is_small_string_substring"] == True)])
                n_total_current = len(df)
                n_fail_current = n_total_current - n_pass_current
                n_pass += n_pass_current
                n_fail += n_fail_current
                n_total += n_total_current
            dict_results[train_or_test] = {
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_total": n_total
            }
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y|%H:%M:%S")
        with open(os.path.join(self.foldername_results, self.dict_foldernames["stats"], f"term_text_filter_stats_{date_time}.json"), "w") as f:
            json.dump(f)
        return dict_results
    
    def normalize_string(self, string):
        return str(string).lower().replace("-", " ")
    
    def generate_features(self, df, to_train=False):
        df["is_acronym"] = df.apply(lambda row: 1 if self.metrics_obj.is_small_string_acronym_long_string(self.normalize_string(row["text"]), self.normalize_string(row["terms"])) else 0, axis=1)
        df["common_tokens"] = df.apply(lambda row: self.metrics_obj.get_common_tokens(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        if to_train:
            # Skip rows that do not have common tokens and acronym relationship
            condition = (len(df["common_tokens"]) == 0) & (df["is_acronym"] == 0)
            df = df[~condition]
            
        df["bigram_similarity"] = df.apply(lambda row: self.metrics_obj.get_ngram_similarity(self.metrics_obj.get_character_ngrams(self.normalize_string(row["text"])), self.metrics_obj.get_character_ngrams(self.normalize_string(row["terms"]))), axis=1)
        df["is_small_string_substring"] = df.apply(lambda row: 1 if self.metrics_obj.is_small_string_substring(row["text"], row["terms"]) else 0, axis=1)
        df["is_pass_filter"] = df.apply(lambda row: 1 if (row["bigram_similarity"] >= 0.5 and row["is_small_string_substring"]) else 0, axis=1)
        df["prefix_combined"] = df.apply(lambda row: self.metrics_obj.get_prefixes(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        df["suffix_combined"] = df.apply(lambda row: self.metrics_obj.get_suffixes(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        df["is_same_numbers"] = df.apply(lambda row: 1 if self.metrics_obj.is_same_numbers(self.normalize_string(row["text"]), self.normalize_string(row["terms"])) else 0, axis=1)
        df["different_tokens"] = df.apply(lambda row: self.metrics_obj.get_different_tokens(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        df["soft_tfidf_similarity"] = df.apply(lambda row: self.metrics_obj.soft_tfidf_similarity(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        df["levenshtein_similarity"] = df.apply(lambda row: self.metrics_obj.levenshtein_similarity(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        df["jaro_winkler_similarity"] = df.apply(lambda row: self.metrics_obj.jaro_winkler_similarity(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        
        return df
    
    def filter_save_term_text_pair_df(self, filepath):
        df = pd.read_csv(filepath, names=self.list_colnames)
        # df["is_acronym"] = df.apply(lambda row: 1 if self.metrics_obj.is_small_string_acronym_long_string(self.normalize_string(row["text"]), self.normalize_string(row["terms"])) else 0, axis=1)
        # df["common_tokens"] = df.apply(lambda row: self.metrics_obj.get_common_tokens(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        # # Skip rows that do not have common tokens and acronym relationship
        # condition = (len(df["common_tokens"]) == 0) & (df["is_acronym"] == 0)
        # df = df[~condition]
        
        # df["bigram_similarity"] = df.apply(lambda row: self.metrics_obj.get_ngram_similarity(self.metrics_obj.get_character_ngrams(self.normalize_string(row["text"])), self.metrics_obj.get_character_ngrams(self.normalize_string(row["terms"]))), axis=1)
        # df["is_small_string_substring"] = df.apply(lambda row: 1 if self.metrics_obj.is_small_string_substring(row["text"], row["terms"]) else 0, axis=1)
        # df["is_pass_filter"] = df.apply(lambda row: 1 if (row["bigram_similarity"] >= 0.5 and row["is_small_string_substring"]) else 0, axis=1)
        # df["prefix_combined"] = df.apply(lambda row: self.metrics_obj.get_prefixes(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        # df["suffix_combined"] = df.apply(lambda row: self.metrics_obj.get_suffixes(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        # df["is_same_numbers"] = df.apply(lambda row: 1 if self.metrics_obj.is_same_numbers(self.normalize_string(row["text"]), self.normalize_string(row["terms"])) else 0, axis=1)
        # df["different_tokens"] = df.apply(lambda row: self.metrics_obj.get_different_tokens(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        # df["soft_tfidf_similarity"] = df.apply(lambda row: self.metrics_obj.soft_tfidf_similarity(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        # df["levenshtein_similarity"] = df.apply(lambda row: self.metrics_obj.levenshtein_similarity(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        # df["jaro_winkler_similarity"] = df.apply(lambda row: self.metrics_obj.jaro_winkler_similarity(self.normalize_string(row["text"]), self.normalize_string(row["terms"])), axis=1)
        
        df = self.generate_features(df)
        df.to_csv(os.path.join(self.foldername_results, self.dict_foldernames["filtered_pairs"], os.path.basename(filepath)), index=False)
        
    def filter_term_text_pairs(self, n_processes=150):
        dict_filepaths = self.load_term_text_pairs_filepaths()
        list_filepaths = dict_filepaths["train"] + dict_filepaths["test"]
        
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = []
            for result in tqdm(pool.imap(self.filter_save_term_text_pair_df, list_filepaths), total=len(list_filepaths)):
                results.append(result)
                
    def split_synonymous(self):
        for train_or_test in tqdm(["train", "test"]):
            for filepath in tqdm(self.load_filtered_filepaths(train_or_test)):
                df = pd.read_csv(filepath)
                df_syn_0 = df[df["is_pass_filter"] == 0]
                df_syn_1 = df[df["is_pass_filter"] == 1]
                df_syn_0.to_csv(path_or_buf=os.path.join(os.path.join(self.foldername_results, "datasets"), f"syn_0_{train_or_test}.csv"), mode="a", header=False)
                df_syn_1.to_csv(path_or_buf=os.path.join(os.path.join(self.foldername_results, "datasets"), f"syn_1_{train_or_test}.csv"), mode="a", header=False)
