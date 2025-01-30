import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from transformers import RobertaTokenizer, RobertaModel
import gc

class GenericTokenizer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(self, text):
        # Tokenize the text and return tokenized tokens as a string collection
        return self.tokenizer.tokenize(text)
    
class BertTokenizerWrapper(GenericTokenizer):
    def __init__(self):
        super().__init__('bert-base-uncased')

class RobertaTokenizerWrapper(GenericTokenizer):
    # def __init__(self):
    #     super().__init__('roberta-base')s
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')        

class GPT2TokenizerWrapper(GenericTokenizer):
    def __init__(self):
        super().__init__('gpt2')
        
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
