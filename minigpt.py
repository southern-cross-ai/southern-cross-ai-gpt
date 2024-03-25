import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import random
import time
import torch
from tqdm import tqdm

"""
@author: Shiqi Ding
@description: This class represents a custom dataset for text data.
             It preprocesses the data for sequence-to-sequence tasks.
@return: Returns preprocessed data for training or evaluation.
"""

class MyDataSet(Data.Dataset):
    def __init__(self, datas):
        """
        Initialize the dataset with provided data.

        Args:
            datas (list): List of input data.
        """
        self.datas = datas

    def __getitem__(self, item):
        """
        Get a single item from the dataset.

        Args:
            item (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing preprocessed data.
                - "decoder_input": Input sequence for decoder.
                - "decoder_input_len": Length of the decoder input.
                - "decoder_output": Output sequence for decoder.
                - "decoder_output_len": Length of the decoder output.
        """
        text_data = self.datas[item]
        text_data = text_data.split("\n")

        text_idx = []
        for data in text_data:
            text_idx.extend([word_2_index.get(i, 1) for i in data])
            text_idx.append(2)
        text_idx.append(2)

        decoder_input = text_idx[:-1]
        decoder_output = text_idx[1:]

        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)

        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.datas)

    def padding_batch(self, batch):
        """
        Pad the batch with "<pad>" tokens to make sequences equal length.

        Args:
            batch (list): List of batch data.

        Returns:
            tuple: Tuple containing padded decoder inputs and decoder outputs.
                - decoder_inputs (Tensor): Padded decoder input sequences.
                - decoder_outputs (Tensor): Padded decoder output sequences.
        """
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]

        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)

        for d in batch:
            d["decoder_input"].extend([word_2_index["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([word_2_index["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)

        return decoder_inputs, decoder_outputs


def make_pad_mask(seq_q, seq_k):
    """
    Generate a padding mask for attention mechanism.

    Args:
        seq_q (Tensor): Query sequence tensor.
        seq_k (Tensor): Key sequence tensor.

    Returns:
        Tensor: Padding mask for attention mechanism.
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def make_lookahead_mask(seq):
    """
    Generate a lookahead mask for attention mechanism.

    Args:
        seq (Tensor): Input sequence tensor.

    Returns:
        Tensor: Lookahead mask for attention mechanism.
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask


class MultiHeadAttention(nn.Module):
    def __init__(self):
        """
        Initialize the Multi-Head Attention layer.

        This module applies multi-head self-attention mechanism.

        """
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, input_x, attn_mask):
        """
        Forward pass of the Multi-Head Attention layer.

        Args:
            input_x (torch.Tensor): Input tensor.
            attn_mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention mechanism.
            torch.Tensor: Attention weights.
        """
        residual, batch_size = input_x, input_x.size(0)
        Q = self.W_Q(input_x).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_x).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_x).view(batch_size, -1,  n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.layernorm(output + residual), attn


class FeedForward(nn.Module):
    def __init__(self):
        """
        Initialize the FeedForward layer.

        This module applies a two-layer feed-forward neural network with ReLU activation.

        """
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),  # Fully connected layer with ReLU activation
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)   # Fully connected layer
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        Forward pass of the FeedForward layer.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after feed-forward operation.
        """
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)  # Residual connection followed by layer normalization


class DecoderBlock(nn.Module):
    def __init__(self):
        """
        Initialize the DecoderBlock.

        This module represents a single block in the decoder of a transformer architecture,
        consisting of multi-head self-attention blocks and feed-forward neural network.

        """
        super(DecoderBlock, self).__init__()
        self.attention_block1 = MultiHeadAttention()
        self.attention_block2 = MultiHeadAttention()
        self.feed_forward = FeedForward()

    def forward(self, dec_inputs, dec_self_attn_mask):
        """
        Forward pass of the DecoderBlock.

        Args:
            dec_inputs (torch.Tensor): Decoder input tensor.
            dec_self_attn_mask (torch.Tensor): Mask for decoder self-attention.

        Returns:
            torch.Tensor: Output tensor after passing through the decoder block.
            torch.Tensor: Decoder self-attention weights.
        """
        # First multi-head self-attention block
        dec_inputs, _ = self.attention_block1.forward(dec_inputs, dec_self_attn_mask)

        # Creating a mask with all zeros
        no_mask = torch.zeros_like(dec_self_attn_mask, device=dec_inputs.device, requires_grad=False)

        # Second multi-head self-attention block
        dec_outputs, dec_self_attn = self.attention_block2.forward(dec_inputs, no_mask)

        # Feed-forward neural network
        dec_outputs = self.feed_forward(dec_outputs)

        return dec_outputs, dec_self_attn


class EmbeddingLayer(nn.Module):
    def __init__(self):
        """
        Initialize the EmbeddingLayer.

        This module represents the embedding layer of a transformer model, which includes
        token embedding and positional encoding.

        """
        super().__init__()

        # Token embedding layer
        self.tgt_emb = nn.Embedding(vocab_size, d_model)

        # Positional embedding layer
        self.pos_emb = nn.Embedding(max_pos, d_model)

    def forward(self, x):
        """
        Forward pass of the EmbeddingLayer.

        Args:
            x (torch.Tensor): Input tensor representing token indices.

        Returns:
            torch.Tensor: Embedded representation of the input tokens with positional encoding added.
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand_as(x)

        # Add positional encoding to token embeddings
        return self.tgt_emb(x) + self.pos_emb(pos)


class Decoder(nn.Module):
    def __init__(self):
        """
        Initialize the Decoder.

        This module represents the decoder part of a transformer model,
        consisting of multiple decoder blocks.

        """
        super(Decoder, self).__init__()

        # Embedding layer
        self.embedding = EmbeddingLayer()

        # Decoder layers
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(n_layers)])

    def forward(self, dec_inputs):
        """
        Forward pass of the Decoder.

        Args:
            dec_inputs (torch.Tensor): Input tensor representing decoder inputs.

        Returns:
            torch.Tensor: Output tensor after passing through the decoder.
            list: List of decoder self-attention weights from each layer.
        """
        # Embedding the decoder inputs
        dec_outputs = self.embedding(dec_inputs)

        # Generating self-attention mask
        dec_self_attn_pad_mask = make_pad_mask(dec_inputs, dec_inputs)  # [b, tgt_len, tgt_len]  Mask off the <pad>.
        dec_self_attn_subsequence_mask = make_lookahead_mask(dec_inputs)  # [b, tgt_len, tgt_len] upper triangular matrix
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)  # [b, tgt_len, tgt_len] All matrices greater than 0 are 1, otherwise 0

        dec_self_attns = []
        # Forward pass through each decoder block
        for layer in self.layers:
            # Forward through decoder block
            # dec_outputs: [b, tgt_len, d_model], dec_self_attn: [b, n_heads, tgt_len, tgt_len], dec_enc_attn: [b, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


class GPT_Model(nn.Module):
    def __init__(self):
        """
        Initialize the GPT_Model.

        This module represents a Generative Pre-trained Transformer model,
        consisting of a decoder, a projection layer, and a loss function.

        """
        super(GPT_Model, self).__init__()

        # Decoder module
        self.decoder = Decoder()

        # Projection layer to predict next token
        self.projection = nn.Linear(d_model, vocab_size)

        # Loss function
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, dec_inputs, label=None):
        """
        Forward pass of the GPT_Model.

        Args:
            dec_inputs (torch.Tensor): Input tensor representing decoder inputs.
            label (torch.Tensor): Target label tensor (optional).

        Returns:
            torch.Tensor: Computed loss if label is provided, else returns logits.
        """
        # Forward pass through the decoder
        dec_outputs, dec_self_attns = self.decoder.forward(dec_inputs)

        # Projecting decoder outputs to vocabulary size
        dec_logits = self.projection(dec_outputs)

        # If labels are provided, compute the loss
        if label is not None:
            loss = self.loss_fun(dec_logits.reshape(-1, dec_logits.shape[-1]), label.reshape(-1))
            return loss
        else:
            # Otherwise, return logits and decoder self-attention weights
            return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    def greedy_decoder(self, dec_input):
        """
        Greedy decoding to generate output sequence.

        Args:
            dec_input (torch.Tensor): Input tensor for decoding.

        Returns:
            torch.Tensor: Output tensor after greedy decoding.
        """
        # Initialization
        terminal = False
        start_dec_len = len(dec_input[0])

        # Greedy decoding loop
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word_2_index['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            dec_outputs, _ = self.decoder(dec_input)
            projected = self.projection(dec_outputs)

            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[-1]
            next_symbol = next_word
            if next_symbol == word_2_index["<sep>"]:
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input

    def random_decoder(self, dec_input, top_n):
        """
        Random decoding to generate output sequence.

        Args:
            dec_input (torch.Tensor): Input tensor for decoding.
            top_n (int): Number of top tokens to consider.

        Returns:
            torch.Tensor: Output tensor after random decoding.
        """
        # Initialization
        terminal = False
        start_dec_len = len(dec_input[0])

        # Random decoding loop
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word_2_index['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            # Forward pass
            with torch.no_grad():
                dec_outputs, _ = self.decoder(dec_input)
                projected = self.projection(dec_outputs)

            a = projected.to('cpu')
            b = a.squeeze(0)[-1]
            c, idx1 = torch.sort(b, descending=True)
            c = np.array(c[:top_n]) ** 2
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
            if next_symbol == word_2_index["<sep>"]:
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input

    def answer(self, sentence):
        """
        Generate an answer for the given input sentence.

        Args:
            sentence (str): Input sentence.

        Returns:
            str: Generated answer.
        """
        # Tokenize input sentence
        dec_input = [word_2_index.get(word, 1) if word != '\t' else word_2_index['<sep>'] for word in sentence]
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0)

        # Greedy decoding to generate output sequence
        output = self.greedy_decoder(dec_input).squeeze(0)

        # Post-processing
        out = [index_2_word[int(id)] for id in output]
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)

        answer = out[sep_indexs[-2] + 1:-1]
        answer = "".join(answer)
        return answer


def read_data(path, num=None):
    """
    Read data from file.

    Args:
        path (str): Path to the file.
        num (int): Number of lines to read (optional).

    Returns:
        list: List of read data.
    """
    with open(path, encoding="utf-8") as f:
        all_data = f.read().split("\n\n")

    if num:
        return all_data[:-1][:num]
    else:
        return all_data[:-1]


if __name__ == "__main__":
    t_data = read_data("data\\train.txt",30)

    with open("data\\vocab.txt",encoding="utf-8") as f :
        index_2_word = f.read().split("\n")
    word_2_index = {w:i for i,w in enumerate(index_2_word) }
    vocab_size = len(word_2_index)
    max_pos = 300
    d_model = 768
    d_ff = 2048
    d_k = d_v = 64
    n_layers = 6
    n_heads = 8
    CLIP = 1
    batch_size = 1
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = MyDataSet(t_data)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

    model = GPT_Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (batch_text, batch_label) in enumerate(tqdm(data_loader)):

            batch_text =  batch_text.to(device)
            batch_label =  batch_label.to(device)
            loss = model.forward(batch_text,batch_label)

            epoch_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

            optimizer.step()
            optimizer.zero_grad()
        end_time = time.time()

        # torch.save(model.state_dict(), r'model1.pt')
        train_loss = epoch_loss
        print(f'\tTrain Loss: {train_loss:.3f}')


    model.eval()
    sentence = ''

    while True:
        sentence = ''
        temp_sentence = input("input:")
        sentence += (temp_sentence + '\t')
        if len(sentence) > 200:
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        print("GPT_Model:", model.answer(sentence))