import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy

import random
import math
import os
from tqdm import tqdm

SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_en = spacy.load('en')

def tokenize_src(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)][:400]

def tokenize_trg(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)][:100]
    
SRC = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
TRG = Field(tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>', lower=True)

home = os.envir['HOME']
data_dir = os.path.join(home, 'data/cnndm')

fields = {'trg': ('trg', TRG), 'src': ('src', SRC)}

train_data, valid_data, test_data = torchtext.data.TabularDataset.splits(
    path = data_dir,
    train = 'cnndm_test.json',
    validation = 'cnndm_test.json',
    test = 'cnndm_test.json',
    format = 'json',
    fields = fields,
)

print(vars(train_data[1]))

SRC.build_vocab(train_data, max_size=50000)
TRG.build_vocab(train_data, max_size=50000)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
     batch_size=BATCH_SIZE,
     sort_key = lambda x : len(x.src),
     sort_within_batch=True,
     device=device)
     
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        
        self.fc_hid = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_out = nn.Linear(hid_dim * 2, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        
        #src = [src sent len, batch size]
        #src_len = [src sent len]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        
        packed_outputs, hidden = self.rnn(packed_embedded)
               
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
        
        #outputs, hidden = self.rnn(embedded)
           
        #outputs = [sent len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
            
        outputs = self.fc_hid(outputs)
            
        #outputs = [sent len, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc_out(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        #outputs = [sent len, batch size, hid dim]
        #hidden = [batch size, hid dim]
        
        return outputs, hidden
        
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.hid_dim = hid_dim

    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, hid dim]
        #encoder_outputs = [src sent len, batch size, hid dim]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(2)
        
        #hidden = [batch size, hid dim, 1]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src sent len, hid dim]
        
        #calculates dot product between hidden and encoder_outputs
        energy = torch.bmm(encoder_outputs, hidden).squeeze(2)
        
        #energy = [batch size, src sent len]
        
        return F.softmax(energy, dim=1) 
   
  class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(hid_dim + emb_dim, hid_dim)
        
        self.out = nn.Linear((hid_dim * 2) + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        #input = [batch size]
        #hidden = [batch size, hid dim]
        #encoder_outputs = [src sent len, batch size, hid dim]
        #mask = [batch size, src sent len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src sent len]
        
        #apply mask
        a = a * mask
        
        #re-normalize attention
        _sums = a.sum(1, keepdim=True)  #keepdim keeps tensor [batch size, 1] instead of making it [batch size]
        a = a.div(_sums) 
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src sent len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src sent len, hid dim]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, hid dim]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, hid dim]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        #rnn_input = [1, batch size, hid dim + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #sent len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        #this also means that output == hidden
        #assert (output == hidden).all(), print(output.shape, hidden.shape, output[0,0,:25], hidden[0,0,:25])
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        
        #output = [bsz, output dim]
        
        return output, hidden.squeeze(0), a.squeeze(1)
  
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.device = device
        
    def create_mask(self, src_len):
        max_len = src_len.max()
        idxs = torch.arange(0,max_len).to(src_len.device)
        mask = (idxs<src_len.unsqueeze(1)).float()
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        
        #src = [src sent len, batch size]
        #src_len = [batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        if trg is None:
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            trg = torch.zeros((20, src.shape[1]), dtype=torch.long).fill_(self.sos_idx).to(src.device)
            
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #tensor to store attention
        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
                
        #first input to the decoder is the <sos> tokens
        output = trg[0,:]
        
        mask = self.create_mask(src_len)
                
        #mask = [batch size, src sent len]
                
        for t in range(1, max_len):
            output, hidden, attention = self.decoder(output, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
            
        return outputs, attentions
        
        
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SOS_IDX = TRG.vocab.stoi['<sos>']

attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, SOS_IDX, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

pad_idx = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(tqdm(iterator)):
        
        src, src_len = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, src_len, trg)
        
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
                
    return epoch_loss / len(iterator)
    
def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src, src_len = batch.src
            trg = batch.trg

            output, _ = model(src, src_len, trg, 0) #turn off teacher forcing

            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
    
N_EPOCHS = 10
CLIP = 10
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'attcnndm_1_model.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
    
    
    
