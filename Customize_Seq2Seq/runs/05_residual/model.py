import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bias=True, residual=True):
        super().__init__()
        self.input_size = input_size            # embedding_dim
        self.hidden_size = hidden_size          # rnn_dim
        self.n_layers = n_layers                # n_layers
        self.layers = nn.ModuleList()
        self.residual = residual

        for i in range(n_layers):               # LSTM 층 쌓기
            self.layers.append(
                nn.LSTMCell(input_size, hidden_size, bias=bias)
            )
            input_size = hidden_size

    def forward(self, inputs, hidden):
        # input => [batch_size, rnn_dim]
        # hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        h_state, c_state = hidden  # 이전 hidden, cell 상태 받아오기

        next_h_state, next_c_state = [], []

        for i, layer in enumerate(self.layers):  # 각 층 layer와 idx
            hi = h_state[i].squeeze(dim=0)
            ci = c_state[i].squeeze(dim=0)
            # squeeze :  차원의 원소가 1인 차원을 모두 없애줌, dim=n : n번째 차원만 1이면 없애줌

            if hi.dim() == 1 and ci.dim() == 1:  # hidden, cell layer의 차원이 1이면
                hi = h_state[i]
                ci = c_state[i]

            next_hi, next_ci = layer(inputs, (hi, ci))
            output = next_hi

            if self.residual:  # 잔차연결
                inputs = output + inputs
            else:
                inputs = output
            next_h_state.append(next_hi)
            next_c_state.append(next_ci)

        next_hidden = (
            torch.stack(next_h_state, dim=0),   # hidden layer concaternate
            torch.stack(next_c_state, dim=0)    # cell layer concaternate
        )
        # input => [batch_size, rnn_dim]
        # next_hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        return inputs, next_hidden


class Recurrent(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, inputs, hidden=None):
        # inputs => [batch_size, sequence_len, embedding_dim]
        # hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        hidden_size = self.cell.hidden_size
        batch_size = inputs.size()[0]

        if hidden is None:
            n_layers = self.cell.n_layers
            zero = inputs.data.new(1).zero_()
            # hidden 초기화
            h0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            # Xavier normal 초기화
            nn.init.xavier_normal_(h0)
            # cell 초기화
            c0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            hidden = (h0, c0)

        outputs = []
        inputs_time = inputs.split(1, dim=1)    # => ([batch_size, 1, embedding_dim] * sequence_len)
        for input_t in inputs_time:             # sequence_len 만큼 반복
            input_t = input_t.squeeze(1)        # => [batch_size, embedding_dim]
            output_t, hidden = self.cell(input_t, hidden)
            outputs += [output_t]

        outputs = torch.stack(outputs, dim=1)
        # outputs => [batch_size, sequence_len, embedding_dim]
        # hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        return outputs, hidden


class Encoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, n_layers=1, residual_used=True):
        super().__init__()
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)

        # rnn cell
        cell = StackLSTMCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim, n_layers=n_layers,
                             bias=rnn_bias, residual=residual_used)

        self.rnn = Recurrent(cell)

    def forward(self, enc_input):
        # enc_input => [batch_size, sequence_len]
        embedded = self.embedding(enc_input)
        # embedded => [batch_size, sequence_len, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded)
        # output => [batch_size, sequence_len, rnn_dim]
        # hidden => [n_layer, batch_size, rnn_dim]
        # cell => [n_layer, batch_size, rnn_dim]
        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, n_layers=1, residual_used=True):
        super().__init__()
        self.vocab_size = embedding_size
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)
        self.hidden_size = rnn_dim
        cell = StackLSTMCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim, n_layers=n_layers,
                             bias=rnn_bias, residual=residual_used)
        self.rnn = Recurrent(cell)
        self.classifier = nn.Linear(rnn_dim, embedding_size)

    def forward(self, dec_input, hidden):
        embedded = self.embedding(dec_input)
        output, hidden = self.rnn(inputs=embedded, hidden=hidden)
        # output => [batch_size, sequence_size, rnn_dim]
        output = self.classifier(output)
        # output => [batch_size, sequence_size, embedding_size]
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, seq_len, beam_search=False, k=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len = seq_len
        self.beam_search = beam_search
        self.k = k
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, enc_input, dec_input, teacher_forcing_rate=0.5):
        # enc_input, dec_input => [batch_size, sequence_len]
        seed_val = 42
        random.seed(seed_val)
        encoder_output, pre_hidden = self.encoder(enc_input)
        # output => [batch_size, sequence_len, rnn_dim]
        # pre_hidden => (hidden, cell)
        # hidden => [n_layer, batch_size, rnn_dim]
        # cell => [n_layer, batch_size, rnn_dim]

        if self.beam_search:
            beam = Beam(self.k, pre_hidden, self.decoder, enc_input.size(0), enc_input.size(1), F.log_softmax,
                        self.device)
            dec_input_i = dec_input[:, 0].unsqueeze(dim=1)
            output = beam.search(dec_input_i, encoder_output)

        else:
            if teacher_forcing_rate == 1.0:  # 교사강요 무조건 적용  => 답을 그대로 다음 input에 넣음
                output, _ = self.decoder(dec_input=dec_input, hidden=pre_hidden)

            else:
                outputs = []
                dec_input_i = dec_input[:, 0].unsqueeze(dim=1)
                for i in range(1, self.seq_len+1):
                    output, pre_hidden = self.decoder(dec_input=dec_input_i, hidden=pre_hidden)
                    _, indices = output.max(dim=2)
                    output = output.squeeze(dim=1)
                    outputs.append(output)

                    if i != self.seq_len:
                        dec_input_i = dec_input[:, i].unsqueeze(dim=1) if random.random() < teacher_forcing_rate \
                            else indices
                output = torch.stack(outputs, dim=1)
            output = F.log_softmax(output, dim=1)

        return output


class Beam:
    r"""
    https://github.com/sh951011/Korean-Speech-Recognition/blob/baf408d14fa1beb88aa50c181630a8878d9f0ba3/models/speller.py#L112
    Applying Beam-Search during decoding process.
    Args:
        k (int) : size of beam
        decoder_hidden (torch.Tensor) : hidden state of decoder
        batch_size (int) : mini-batch size during infer
        max_len (int) :  a maximum allowed length for the sequence to be processed
        function (torch.nn.Module) : A function used to generate symbols from RNN hidden state
        (default : torch.nn.functional.log_softmax)
        decoder (torch.nn.Module) : get pointer of decoder object to get multiple parameters at once
        beams (torch.Tensor) : ongoing beams for decoding
        probs (torch.Tensor) : cumulative probability of beams (score of beams)
        sentences (list) : store beams which met <eos> token and terminated decoding process.
        sentence_probs (list) : score of sentences
    Inputs: decoder_input, encoder_outputs
        - **decoder_input** (torch.Tensor): initial input of decoder - <sos>
        - **encoder_outputs** (torch.Tensor): tensor with containing the outputs of the encoder.
    Returns: y_hats
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
    Examples::
        # >>> beam = Beam(k, decoder_hidden, decoder, batch_size, max_len, F.log_softmax)
        # >>> y_hats = beam.search(inputs, encoder_outputs)
    """

    def __init__(self, k, decoder_hidden, decoder, batch_size, max_len, function, device):
        assert k > 1, "beam size (k) should be bigger than 1"
        self.k = k
        self.device = device
        self.decoder_hidden = decoder_hidden
        self.batch_size = batch_size
        self.max_len = max_len
        self.function = function
        self.rnn = decoder.rnn
        self.embedding = decoder.embedding
        self.input_dropout = decoder.embedding_dropout
        self.use_attention = None
        self.attention = None
        self.hidden_size = decoder.hidden_size
        self.vocab_size = decoder.vocab_size
        self.w = nn.Linear(self.hidden_size, self.vocab_size).to(self.device)
        self.eos_id = 1
        self.beams = None
        self.probs = None
        self.sentences = [[] for _ in range(self.batch_size)]
        self.sentence_probs = [[] for _ in range(self.batch_size)]

    def search(self, decoder_input, encoder_outputs):
        # decoder_input => [batch_size, 1]
        # encoder_outputs => [ batch_size, seq_len, hidden_size]

        # get class classfication distribution => [batch_size, vocab_size]
        step_outputs = self._forward_step(decoder_input, encoder_outputs).squeeze(1)

        # get top K probability & idx => probs =[batch_size, k], beams = [batch_size, k]
        self.probs, self.beams = step_outputs.topk(self.k)
        decoder_input = self.beams
        # transpose => [batch_size, k, 1]
        self.beams = self.beams.view(self.batch_size, self.k, 1)
        for di in range(self.max_len-1):
            if self._is_done():
                break
            # For each beam, get class classfication distribution (shape: BxKxC)
            predicted_softmax = self._forward_step(decoder_input, encoder_outputs)
            step_output = predicted_softmax.squeeze(1)
            # get top k distribution (shape: BxKxK)
            child_ps, child_vs = step_output.topk(self.k)
            # get child probability (applying length penalty)
            child_ps = self.probs.view(self.batch_size, 1, self.k) + child_ps
            child_ps /= self._get_length_penalty(length=di+1, alpha=1.2, min_length=5)
            # Transpose (BxKxK) => (BxK^2)
            child_ps = child_ps.view(self.batch_size, self.k * self.k)
            child_vs = child_vs.view(self.batch_size, self.k * self.k)
            # Select Top k in K^2 (shape: BxK)
            topk_child_ps, topk_child_ids = child_ps.topk(self.k)
            # Initiate topk_child_vs (shape: BxK)
            topk_child_vs = torch.LongTensor(self.batch_size, self.k)
            # Initiate parent_beams (shape: BxKxS)
            parent_beams = torch.LongTensor(self.beams.size())
            # ids // k => ids of topk_child`s parent node
            parent_beams_ids = (topk_child_ids // self.k).view(self.batch_size, self.k)

            for batch_num, batch in enumerate(topk_child_ids):
                for beam_idx, topk_child_idx in enumerate(batch):
                    topk_child_vs[batch_num, beam_idx] = child_vs[batch_num, topk_child_idx]
                    parent_beams[batch_num, beam_idx] = self.beams[batch_num, parent_beams_ids[batch_num, beam_idx]]
            # append new_topk_child (shape: BxKx(S) => BxKx(S+1))
            self.beams = torch.cat([parent_beams, topk_child_vs.view(self.batch_size, self.k, 1)], dim=2)
            self.probs = topk_child_ps

            if torch.any(topk_child_vs == self.eos_id):
                done_ids = torch.where(topk_child_vs == self.eos_id)
                count = [1] * self.batch_size # count done beams
                for (batch_num, beam_idx) in zip(*done_ids):
                    self.sentences[batch_num].append(self.beams[batch_num, beam_idx])
                    self.sentence_probs[batch_num].append(self.probs[batch_num, beam_idx])
                    self._replace_beam(
                        child_ps=child_ps,
                        child_vs=child_vs,
                        done_ids=(batch_num, beam_idx),
                        count=count[batch_num]
                    )
                    count[batch_num] += 1
            # update decoder_input by topk_child_vs
            decoder_input = topk_child_vs
        y_hats = self._get_best()
        return y_hats

    def _get_best(self):
        """ get sentences which has the highest probability at each batch, stack it, and return it as 2d torch """
        y_hats = []

        for batch_num, batch in enumerate(self.sentences):
            if len(batch) == 0:
                # if there is no terminated sentences, bring ongoing sentence which has the highest probability instead
                prob_batch = self.probs[batch_num].to(self.device)
                top_beam_idx = int(prob_batch.topk(1)[1])
                y_hats.append(self.beams[batch_num, top_beam_idx])

            else:
                # bring highest probability sentence
                top_beam_idx = int(torch.FloatTensor(self.sentence_probs[batch_num]).topk(1)[1])
                y_hats.append(self.sentences[batch_num][top_beam_idx])
        y_hats = self._match_len(y_hats).to(self.device)
        return y_hats

    def _match_len(self, y_hats):
        max_len = -1
        for y_hat in y_hats:
            if len(y_hat) > max_len:
                max_len = len(y_hat)

        matched = torch.LongTensor(self.batch_size, max_len).to(self.device)
        for batch_num, y_hat in enumerate(y_hats):
            matched[batch_num, :len(y_hat)] = y_hat
            matched[batch_num, len(y_hat):] = 0

        return matched

    def _is_done(self):
        """ check if all beam search process has terminated """
        for done in self.sentences:
            if len(done) < self.k:
                return False
        return True

    def _forward_step(self, decoder_input, encoder_outputs):
        """ forward one step on each decoder cell """
        decoder_input = decoder_input.to(self.device)
        output_size = decoder_input.size(1)
        embedded = self.embedding(decoder_input).to(self.device)
        embedded = self.input_dropout(embedded)
        decoder_output, hidden = self.rnn(inputs=embedded, hidden=self.decoder_hidden)  # decoder output

        if self.use_attention:
            output = self.attention(decoder_output, encoder_outputs)
        else:
            output = decoder_output
        predicted_softmax = self.function(self.w(output.contiguous().view(-1, self.hidden_size)), dim=1).to(self.device)
        predicted_softmax = predicted_softmax.view(self.batch_size, output_size, -1)
        return predicted_softmax

    def _get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((min_length + length) / (min_length + 1)) ** alpha

    def _replace_beam(self, child_ps, child_vs, done_ids, count):
        """ Replaces a beam that ends with <eos> with a beam with the next higher probability. """
        done_batch_num, done_beam_idx = done_ids[0], done_ids[1]
        tmp_ids = child_ps.topk(self.k + count)[1]
        new_child_idx = tmp_ids[done_batch_num, -1]
        new_child_p = child_ps[done_batch_num, new_child_idx].to(self.device)
        new_child_v = child_vs[done_batch_num, new_child_idx].to(self.device)
        parent_beam_idx = (new_child_idx // self.k)
        parent_beam = self.beams[done_batch_num, parent_beam_idx].to(self.device)
        parent_beam = parent_beam[:-1]
        new_beam = torch.cat([parent_beam, new_child_v.view(1)])
        self.beams[done_batch_num, done_beam_idx] = new_beam
        self.probs[done_batch_num, done_beam_idx] = new_child_p