"""
Code related to string similarity metrics
"""
import numpy as np
import re
from py_stringmatching.similarity_measure.soft_tfidf import SoftTfIdf
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.levenshtein import Levenshtein

def exception_handler(func):
    def inner_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            return None
    return inner_func

class Metrics:
    def get_character_ngrams(self, phrase, n=2):
        phrase = str(phrase)
        return [phrase[i: i + n] for i in range(len(phrase) - n + 1)]
    
    def get_ngram_similarity(self, list_ngrams_a, list_ngrams_b, case_sensitive=False):
        if not case_sensitive:
            list_ngrams_a, list_ngrams_b = [[str(ngram).lower() for ngram in l] for l in [list_ngrams_a, list_ngrams_b]]
        ngrams_a = set(list_ngrams_a)
        ngrams_b = set(list_ngrams_b)
        try:
            ngram_similarity = (2 * len(ngrams_a.intersection(ngrams_b)))/(len(ngrams_a) + len(ngrams_b))
        except ZeroDivisionError:
            ngram_similarity = np.inf
        except Exception as e:
            print(e)
            print(list_ngrams_a, list_ngrams_b)
        return ngram_similarity
    
    def is_small_string_substring(self, string_a, string_b, case_sensitive=False):
        string_a, string_b = str(string_a), str(string_b)
        if not case_sensitive:
            string_a, string_b = str(string_a).lower(), str(string_b).lower()
        string_longer, string_shorter = str(string_b), str(string_a)
        if len(string_a) > len(string_b):
            string_longer = string_a
            string_shorter = string_b
        string_longer = "".join([char for char in string_longer if char in string_shorter])
        string_longer, string_shorter = [[char for char in string if char != " "] for string in [string_longer, string_shorter]]
        string_longer, string_shorter = "".join(string_longer), "".join(string_shorter)
        return True if string_longer == string_shorter else False
    
    def get_common_ngrams(self, string_a, string_b, n=2):
        string_a, string_b = str(string_a), str(string_b)
        return list(set(self.get_character_ngrams(string_a, n=n)).intersection(self.get_character_ngrams(string_b, n=n)))
        
    def get_prefixes(self, string_a, string_b, n=3):
        string_a, string_b = str(string_a), str(string_b)
        return string_a.lower()[:n], string_b.lower()[:n]

    def get_suffixes(self, string_a, string_b, n=3):
        string_a, string_b = str(string_a), str(string_b)
        return string_a.lower()[-n:], string_b.lower()[-n:]
    
    def is_same_numbers(self, string_a, string_b):
        string_a, string_b = str(string_a), str(string_b)
        list_num_a, list_num_b = [[char for char in string if char.isdigit()] for string in [string_a, string_b]]
        set_a, set_b = set(list_num_a), set(list_num_b)
        return set_a == set_b
    
    def is_small_string_acronym_long_string(self, string_a, string_b, case_sensitive=False):
        return self.is_small_string_substring(string_a, string_b, case_sensitive=case_sensitive)

    def generate_tokens(self, string_a, string_b):
        try:
            regex_str = "- |\/ |( |) |[ |],"
            tokens_a, tokens_b = [[token for token in list_tokens if token != ""] for list_tokens in [re.split(regex_str, str(string_a)), re.split(regex_str, str(string_b))]]
            return set(tokens_a), set(tokens_b)
        except Exception:
            return set(), set()
    
    def get_common_tokens(self, string_a, string_b):
        tokens_a, tokens_b = self.generate_tokens(string_a, string_b)
        return tokens_a.intersection(tokens_b)
    
    def get_different_tokens(self, string_a, string_b):
        tokens_a, tokens_b = self.generate_tokens(string_a, string_b)
        return tokens_a.symmetric_difference(tokens_b)
    
    def soft_tfidf_similarity(self, string_a, string_b, threshold=0.9):
        try:
            string_a, string_b = str(string_a), str(string_b)
            tokens_a, tokens_b = [list(tokens) for tokens in self.generate_tokens(string_a, string_b)]
            return SoftTfIdf(sim_func=JaroWinkler().get_raw_score, threshold=threshold).get_raw_score(tokens_a, tokens_b)
        except Exception:
            return None
    
    def levenshtein_similarity(self, string_a, string_b):
        string_a, string_b = str(string_a), str(string_b)
        return Levenshtein().get_sim_score(string_a, string_b)
    
    def jaro_winkler_similarity(self, string_a, string_b):
        string_a, string_b = str(string_a), str(string_b)
        return JaroWinkler().get_sim_score(string_a, string_b)

if __name__ == "__main__":
    metrics_obj = Metrics()
    string_a, string_b = "GATA binding protein 5", "GATA binding factor 5"
    print(metrics_obj.soft_tfidf_similarity(string_a, string_b))