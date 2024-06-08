from collections import defaultdict
import os
import pandas as pd
from PIL import Image
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


frequency_threshold = 5
batch_size = 2


class Vocabulary:
    def __init__(self, frequency_threshold, texts):
        self.stoi = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.itos = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}

        self.frequency_threshold = frequency_threshold
        self.spacy_eng = spacy.load('en_core_web_sm')

        self.build_vocabulary(texts)

    def tokenize(self, text):
        tokenized_text = [
            token.text.lower() for token in self.spacy_eng.tokenizer(text)
        ]
        return tokenized_text

    def build_vocabulary(self, texts):
        frequencies = defaultdict(int)
        index = 4

        for text in texts:
            for token in self.tokenize(text):
                frequencies[token] += 1
                if frequencies[token] == self.frequency_threshold:
                    self.stoi[token] = index
                    self.itos[index] = token
                    index += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        numericalized_text = [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]
        return numericalized_text

    def denumericalize(self, numericalized_text):
        text = [self.itos[token.item()] for token in numericalized_text]
        return text


class Collate:
    def __init__(self, padding_value):
        self.padding_value = padding_value

    def __call__(self, batch):
        images = [pair[0].unsqueeze(0) for pair in batch]
        images = torch.cat(images, dim=0)

        captions = [pair[1] for pair in batch]
        captions = pad_sequence(
            captions, batch_first=True, padding_value=self.padding_value,
        )

        return images, captions


class FlickerDataset(Dataset):
    def __init__(
        self, images_path, captions_path, frequency_threshold, transform=None,
    ):
        self.df = pd.read_csv(captions_path)
        self.images_index = self.df['image']
        self.captions = self.df['caption']

        self.images_path = images_path
        self.transform = transform
        self.vocab = Vocabulary(frequency_threshold, self.captions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_index = self.images_index[index]
        image = Image.open(
            os.path.join(self.images_path, image_index)
        ).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption = self.captions[index]
        numericalized_caption = [self.vocab.stoi['<SOS>']]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi['<EOS>'])
        numericalized_caption = torch.tensor(numericalized_caption)

        return image, numericalized_caption

    def translate(self, numericalized_text):
        return self.vocab.denumericalize(numericalized_text)


def get_dataloader():
    images_path = 'data/images'
    captions_path = 'data/captions.txt'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = FlickerDataset(
        images_path, captions_path, frequency_threshold, transform
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=Collate(dataset.vocab.stoi['<PAD>']), shuffle=True,
        num_workers=11, persistent_workers=True, pin_memory=True,
    )

    print()
    print("Data Loading Completed!")
    print()
    images, captions = next(iter(dataloader))
    print(f"len(dataset) = {len(dataset)}")
    print(f"len(dataloader) = {len(dataloader)}")
    print(f"batch_size = {batch_size}")
    print(f"images.shape = {images.shape}")
    print(f"captions.shape = {captions.shape}")
    print(f"captions[0] = {captions[0]}")
    print(f"dataset.translate(captions[0]) = {dataset.translate(captions[0])}")
    print("Data Validation Completed!")
    print()

    return dataloader


if __name__ == '__main__':
    get_dataloader()
