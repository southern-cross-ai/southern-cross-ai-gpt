
import torch
import json
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import random
import time
import torch
from tqdm import tqdm

def get_attn_pad_mask(seq_q, seq_k): 

    batch_size, len_q = seq_q.size() 
    batch_size, len_k = seq_k.size()
  
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] 
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask 

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask): 

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  
        scores.masked_fill_(attn_mask, -1e9)  
        

        attn = nn.Softmax(dim=-1)(scores) 
        context = torch.matmul(attn, V)  
        return context, attn

class MultiHeadAttention(nn.Module): 
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask): 

        residual, batch_size = input_Q, input_Q.size(0)  
        
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  
        V = self.W_V(input_V).view(batch_size, -1,  n_heads, d_v).transpose(1, 2)  

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  
        output = self.fc(context)  
        return self.layernorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):   
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)  

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention() 
        
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask): 

        
        dec_outputs, dec_self_attn = self.dec_self_attn.forward(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs)  
        return dec_outputs, dec_self_attn  

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)  
        self.pos_emb = nn.Embedding(max_pos, d_model)     
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs): 

        seq_len = dec_inputs.size(1) 
        pos = torch.arange(seq_len, dtype=torch.long, device=device) 
        pos = pos.unsqueeze(0).expand_as(dec_inputs)  

        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)  
        

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)  

        dec_self_attns = []
        for layer in self.layers:
            
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns

class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocab_size) 

    def forward(self, dec_inputs): 
        dec_outputs, dec_self_attns = self.decoder.forward(dec_inputs)  
        dec_logits = self.projection(dec_outputs)  
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    def greedy_decoder(self, dec_input):

        terminal = False
        start_dec_len = len(dec_input[0])

        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break


            dec_outputs, _ = self.decoder(dec_input)
            projected = self.projection(dec_outputs)

            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            if next_symbol == word2id["<sep>"]:
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input

    
    def random_decoder(self, dec_input,top_n): 

        terminal = False
        start_dec_len = len(dec_input[0])
        
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach() , torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            
            with torch.no_grad():
                dec_outputs, _ = self.decoder(dec_input)
                projected = self.projection(dec_outputs) 

            
            

            a = projected.to('cpu') 
            b = a.squeeze(0)[-1]  
            c, idx1 = torch.sort(b, descending=True)  
            c = np.array(c[:top_n])**2  
            idx1 = np.array(idx1[:top_n])


            sum = 0 
            for i in c:
                sum += i 

            d = sum * random.uniform(0, 1) 

            for i, j in enumerate(c):
                d -= j  
                if d <= 0:
                    next_word = idx1[i] 
                    break

            next_symbol = next_word
            if next_symbol == word2id["<sep>"]: 
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input   

    def answer(self, sentence): 
        
        dec_input = [word2id.get(word, 1) if word != '\t' else word2id['<sep>'] for word in sentence]  
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0) 

        output = self.greedy_decoder(dec_input).squeeze(0)  
        

        out = [id2word[int(id)] for id in output]  
        
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)

        
        answer = out[sep_indexs[-2] + 1:-1]

        answer = "".join(answer)
        return answer

if __name__ == '__main__':

    with open("data/vocab.txt",encoding="utf-8") as f :
        id2word = f.read().split("\n")
    word2id = {w:i for i,w in enumerate(id2word) }
    vocab_size = len(word2id)
    max_pos = 300
    d_model = 768
    d_ff = 2048
    d_k = d_v = 64
    n_layers = 6
    n_heads = 8


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT().to(device)
    model.load_state_dict(torch.load('model.pt',map_location=device))


    model.eval()
    sentence = ''
    while True:
        sentence = ''
        temp_sentence = input("please input:")
        sentence += (temp_sentence + '\t')
        if len(sentence) > 200:
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        print("miniGPT:", model.answer(sentence))

