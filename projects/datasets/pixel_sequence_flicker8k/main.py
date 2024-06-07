import os
import pandas as pd
from PIL import Image
import spacy
import spacy.tokenizer
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class Vocabulary:
    def __init__(self, frequency):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.frequency = frequency

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy.tokenizer(text)]

    def build_vocabulary(self, sentences):
        frequencies = defaultdict(int)
        idx = 4

        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.frequency:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, caption):
        tokenized_text = self.tokenizer_eng(caption)
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, frequency=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(frequency)
        self.vocab.build_vocabulary(self.captions.to_list())

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        caption = self.captions[index]
        numericalized_caption = [self.vocab.stoi['<SOS>']]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi['<EOS>'])

        return img, torch.tensor(numericalized_caption)
