import torch
import torch.nn as nn
import torchtext
import pandas as pd
# from torchtext import data
# from torchtext.legacy import data
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data
else:
    from torchtext.legacy import data

from tqdm.notebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNNClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        # word_vector,
        hidden_size,
        in_emb_size,
        emb_size,
        n_classes,
        n_layers=4,
        dropout_rate=0.3,
    ):
        self.input_size = input_size
        # self.word_vector = word_vector
        self.hidden_size = hidden_size
        self.in_emb_size = in_emb_size
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        self.emb = nn.Embedding(self.in_emb_size, self.emb_size)
        
        self.layer1 = nn.RNN(self.input_size, self.hidden_size)
        self.layer2 = nn.RNN(self.hidden_size, self.hidden_size)
        self.layer3 = nn.RNN(self.hidden_size, self.hidden_size)
        self.layer4 = nn.RNN(self.hidden_size, self.n_classes)
        self.actv = nn.Softmax()
        
        #구현
    
    def forward(self, x):
        x_emb = []
        
        for x_s in x:
            x_emb.append(self.emb(x_s))

        x1 = self.layer1(x_emb)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        ans = self.actv(x4)

        return ans
        #구현        

class DataLoader(object):

    def __init__(
        self, pathss,
        batch_size=64,
        valid_ratio=.2,
        device=-1,
        max_vocab=50000,
        min_freq=1,
        use_eos=False,
        shuffle=True,
    ):

        super().__init__()
        
        self.label = data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None
        )
        
        self.text = data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
            eos_token='<EOS>' if use_eos else None,
        )
        
        train, valid = data.TabularDataset(
            path=pathss,
            format='csv', 
            fields=[
                ('label', self.label),
                ('text', self.text),
            ],
        ).split(split_ratio=(1 - valid_ratio))

        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:0' ,
            shuffle=shuffle,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
        )

        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq)


dataLoad = DataLoader('J:/내 드라이브/Young/Study/Transformer/final_combine.csv')

label = dataLoad.label
text = dataLoad.text

train_loader = dataLoad.train_loader
val_loader = dataLoad.valid_loader

vars(label.vocab)
print(len(label.vocab.stoi))
try:
    label.vocab.stoi.pop('label')
except:
    pass
print(len(label.vocab.stoi))

vars(text.vocab)
print(len(text.vocab.stoi))
try:
    text.vocab.stoi.pop('text')
except:
    pass
print(len(text.vocab.stoi))

epoch = 100
lr = 0.001

emb_size = 512
hidden_size = 512
n_words = len(text.vocab.stoi)
n_cls = len(label.vocab.stoi)


print(emb_size)
print(hidden_size)
print(n_words)
print(n_cls)

model = RNNClassifier(input_size=emb_size, hidden_size=hidden_size, in_emb_size=n_words, emb_size = emb_size, n_classes=n_cls) # word_vector=text.vocab.stoi,


model.to(device)
model.train()
crit = nn.NLLLoss()
optim = torch.optim.Adam(model.parameters(), lr = lr)
