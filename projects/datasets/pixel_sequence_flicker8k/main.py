from collections import defaultdict
import os
import pandas as pd
from PIL import Image
import spacy
import spacy.tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


spacy_eng = spacy.load('en_core_web_sm')


class Vocabulary:
    def __init__(self, frequency):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.frequency = frequency

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

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


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(
            targets, batch_first=True, padding_value=self.pad_idx,
        )
        return imgs, targets


def get_loader(
    root_dir, captions_file, transform, batch_size=32, num_workers=11,
    pin_memory=True,
):
    dataset = FlickerDataset(root_dir, captions_file, transform)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=dataset.vocab.stoi['<PAD>']),
    )
    return dataloader


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataloader = get_loader('data/images', 'data/captions.txt', transform)

    imgs, captions = next(iter(dataloader))
    print(f"len(dataloader) = {len(dataloader)}")
    print(f"imgs.shape = {imgs.shape}")
    print(f"captions.shape = {captions.shape}")


if __name__ == '__main__':
    main()
