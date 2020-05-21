import os
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers import BertWordPieceTokenizer


class MLMDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, file_path: str, args):
        print(file_path)
        assert os.path.isfile(file_path)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.bert_model_type +
            "_cached_mlm_" + filename
        )

        if os.path.exists(cached_features_file):
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.samples = torch.load(handle)
        else:
            print("Creating features from dataset file at %s", directory)

            # Get the faster tokenizer from tokenizers package
            tokenizer.save_vocabulary(vocab_path='.')
            fast_tokenizer = BertWordPieceTokenizer(
                "vocab.txt", lowercase=args.lowercase)
            fast_tokenizer.enable_truncation(tokenizer.max_len)
            fast_tokenizer.enable_padding(
                max_length=tokenizer.max_len, pad_token=tokenizer.pad_token)

            self.samples = []

            # Load data over here
            df = pd.read_json(file_path)
            print('SQUAD data: ')

            for _, row in tqdm(df.iterrows(), total=df.shape[0]):
                for paragraph in row['data']['paragraphs']:
                    context = paragraph['context']
                    for qa_pair in paragraph['qas']:
                        question = qa_pair['question']

                        batch = fast_tokenizer.encode(question, context)
                        self.samples.append({
                            'input_ids': batch.ids,
                            'attention_mask': batch.attention_mask
                        })

                        for encoding in batch.overflowing:
                            self.samples.append({
                                'input_ids': encoding.ids,
                                'attention_mask': encoding.attention_mask
                            })

            df = None

            print("Saving features into cached file: ", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                torch.save(self.samples, handle,
                           pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return {
            'input_ids': torch.tensor(self.samples[i]['input_ids']),
            'attention_mask': torch.tensor(self.samples[i]['attention_mask'])
        }
