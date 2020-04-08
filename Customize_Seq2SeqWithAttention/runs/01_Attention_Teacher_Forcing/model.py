import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class StackLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bias=True):
        super().__init__()
        self.input_size = input_size            # embedding_dim
        self.hidden_size = hidden_size          # rnn_dim
        self.n_layers = n_layers                # n_layers
        self.layers = nn.ModuleList()

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

    def forward(self, inputs, pre_hidden=None, get_attention=False, attention=None, encoder_outputs=None):
        # inputs => [batch_size, sequence_len, embedding_dim]
        # hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        hidden_size = self.cell.hidden_size
        batch_size = inputs.size()[0]

        if pre_hidden is None:
            n_layers = self.cell.n_layers
            zero = inputs.data.new(1).zero_()
            # hidden 초기화
            h0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            # cell 초기화
            c0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            hidden = (h0, c0)
        else:
            hidden = pre_hidden
        outputs = []
        attentions = []
        inputs_time = inputs.split(1, dim=1)    # => ([batch_size, 1, embedding_dim] * sequence_len)
        for input_t in inputs_time:             # sequence_len 만큼 반복
            input_t = input_t.squeeze(1)        # => [batch_size, embedding_dim]
            last_hidden_t, hidden = self.cell(input_t, hidden)
            if get_attention:
                output_t, score = attention(encoder_outputs, last_hidden_t)
                attentions.append(score)
            outputs += [last_hidden_t]

        outputs = torch.stack(outputs, dim=1)
        # outputs => [batch_size, sequence_len, embedding_dim]
        # hidden => (h_state, c_state)
        # h_state, c_state = [n_layers, batch_size, hidden_size]
        if get_attention:
            attentions = torch.stack(attentions, dim=2)
            return outputs, hidden, attentions
        return outputs, hidden


class Encoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)    # Embedding Layer

        # rnn cell
        cell = StackLSTMCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim, n_layers=n_layers,
                             bias=rnn_bias)

        self.rnn = Recurrent(cell)

    def forward(self, enc_input):
        # enc_input => [batch_size, sequence_len]
        embedded = self.embedding(enc_input)
        embedded = F.relu(embedded)
        # embedded => [batch_size, sequence_len, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded)
        # output => [batch_size, sequence_len, rnn_dim]
        # hidden => [n_layer, batch_size, rnn_dim]
        # cell => [n_layer, batch_size, rnn_dim]
        return output, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size, score_function):
        super().__init__()
        if score_function not in ['dot', 'general', 'concat']:
            raise NotImplemented('Not implemented {} attention score function '
                                 'you must selected [dot, general, concat]'.format(score_function))

        self.character_distribution = nn.Linear(hidden_size * 2, hidden_size)
        self.score_function = score_function
        if score_function == 'dot':
            pass
        elif score_function == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            raise NotImplementedError

    def forward(self, context, target):
        # context => [batcxh_size, seq_len, hidden]
        # target => [batcxh_size, hidden]
        # batch_size, _ = context.size()
        batch_size, seq_len, _ = context.size()
        if self.score_function == 'dot':
            x = target.unsqueeze(-1)
            attention_weight = context.bmm(x).squeeze(-1)

        elif self.score_function == 'general':
            x = self.Wa(target)
            x = x.unsqueeze(-1)
            attention_weight = context.bmm(x).squeeze(-1)       # => [batch_size, seq_len)
        else:
            raise NotImplementedError

        attention_distribution = F.softmax(attention_weight, -1)    # => [batch_size, seq_len]
        context_vector = attention_distribution.unsqueeze(1).bmm(context).squeeze(1)
        # [batch size, 1, seq_len] * [batch_size, seq_len, hidden] = [batch_size, hidden]
        combine = self.character_distribution(torch.cat((context_vector, target), 1))

        return combine, attention_distribution


class AttentionDecoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, n_layers=1,
                 attention_score_func='dot'):
        super().__init__()
        self.vocab_size = embedding_size
        self.hidden_size = rnn_dim              # beam search 적용시 사용하는 변수
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)
        self.attention = Attention(hidden_size=rnn_dim, score_function=attention_score_func)
        cell = StackLSTMCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim,
                             n_layers=n_layers, bias=rnn_bias)
        self.rnn = Recurrent(cell)                              # 기본 rnn
        self.classifier = nn.Linear(rnn_dim, embedding_size)    # dense

    def forward(self, dec_input, hidden, encoder_outputs, get_attention=False):
        # dec_intput => [batch_size, seq_len]
        # encoder_outputs => [batch_size, seq_len, hidden]
        # hidden[0] => [n_layers, batch_size, hidden]
        embedded = self.embedding(dec_input)
        embedded = F.relu(embedded)
        if get_attention:
            output, hidden, attention_score = self.rnn(inputs=embedded, pre_hidden=hidden, get_attention=get_attention,
                                                       attention=self.attention, encoder_outputs=encoder_outputs)
            output = self.classifier(output)
            # output => [batch_size, sequence_size, embedding_size]
            return output, hidden, attention_score

        else:
            output, hidden = self.rnn(inputs=embedded, pre_hidden=hidden)
            # output => [batch_size, sequence_size, rnn_dim]
            output = self.classifier(output)  # dense 라인 적용
            # output => [batch_size, sequence_size, embedding_size]
            return output, hidden


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, seq_len, get_attention, beam_search=False, k=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len = seq_len
        self.get_attention = get_attention
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
            if self.get_attention:
                beam = Beam(self.k, pre_hidden, self.decoder, enc_input.size(0), enc_input.size(1), F.log_softmax,
                            self.device, get_attention=self.get_attention)
            else:
                beam = Beam(self.k, pre_hidden, self.decoder, enc_input.size(0), enc_input.size(1), F.log_softmax,
                            self.device)
            dec_input_i = dec_input[:, 0].unsqueeze(dim=1)

            output = beam.search(dec_input_i, encoder_output)
        else:
            # teacher forcing ratio check
            if teacher_forcing_rate == 1.0:  # 교사강요 무조건 적용  => 답을 그대로 다음 input에 넣음
                if self.get_attention:
                    output, _, attentions = self.decoder(encoder_outputs=encoder_output, dec_input=dec_input,
                                                         hidden=pre_hidden, get_attention=True)
                else:
                    output, _ = self.decoder(encoder_outputs=encoder_output, dec_input=dec_input,
                                             hidden=pre_hidden, get_attention=False)
            else:
                outputs = []
                dec_input_i = dec_input[:, 0].unsqueeze(dim=1)
                if self.get_attention:
                    attentions = []
                    for i in range(1, self.seq_len + 1):
                        output, pre_hidden, attention = self.decoder(encoder_outputs=encoder_output,
                                                                     dec_input=dec_input_i, hidden=pre_hidden,
                                                                     get_attention=True)
                        _, indices = output.max(dim=2)

                        output = output.squeeze(dim=1)
                        attention = attention.squeeze(dim=2)
                        outputs.append(output)
                        attentions.append(attention)

                        if i != self.seq_len:
                            dec_input_i = \
                                dec_input[:, i].unsqueeze(dim=1) if random.random() < teacher_forcing_rate else indices

                    output = torch.stack(outputs, dim=1)
                    attentions = torch.stack(attentions, dim=2)
                else:
                    for i in range(1, self.seq_len + 1):
                        output, pre_hidden = self.decoder(encoder_outputs=encoder_output, dec_input=dec_input_i,
                                                          hidden=pre_hidden,  get_attention=False)
                        _, indices = output.max(dim=2)
                        output = output.squeeze(dim=1)
                        outputs.append(output)

                        if i != self.seq_len:
                            dec_input_i = \
                                dec_input[:, i].unsqueeze(dim=1) if random.random() < teacher_forcing_rate else indices

                    output = torch.stack(outputs, dim=1)
            output = F.log_softmax(output, dim=1)
            if self.get_attention:
                return output, attentions
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

    def __init__(self, k, decoder_hidden, decoder, batch_size, max_len, function, device, get_attention=False):
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
        self.use_attention = get_attention
        if get_attention:
            self.attention = decoder.attention
        else:
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
        # 상위 k개 뽑기
        self.probs, self.beams = step_outputs.topk(self.k)
        decoder_input = self.beams
        # transpose => [batch_size, k, 1]
        self.beams = self.beams.view(self.batch_size, self.k, 1)
        for di in range(self.max_len - 1):
            if self._is_done():
                break
            # For each beam, get class classfication distribution (shape: BxKxC)
            # 현재 시점에서 확률벡터를 구함
            predicted_softmax = self._forward_step(decoder_input, encoder_outputs)
            step_output = predicted_softmax.squeeze(1)
            # get top k distribution (shape: BxKxK)
            # 현재 확률벡터 기준으로 상위벡터 k개를 구함
            child_ps, child_vs = step_output.topk(self.k)
            # get child probability (applying length penalty)
            # length penalty
            child_ps = self.probs.view(self.batch_size, 1, self.k) + child_ps
            child_ps /= self._get_length_penalty(length=di + 1, alpha=1.2, min_length=5)
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
            self.beams = torch.cat([parent_beams, topk_child_vs.view(self.batch_size, self.k, 1)], dim=2).to(self.device)
            self.probs = topk_child_ps.to(self.device)

            if torch.any(topk_child_vs == self.eos_id):
                done_ids = torch.where(topk_child_vs == self.eos_id)    # eos id 가 나오면 done_ids에 저장
                count = [1] * self.batch_size                           # count done beams
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
        """ 최종후보 k개 중에서 최고 확률이 높은 1개를 선택해서 출력 """
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
        """ 만약에 y_hat이 sequence_len보다 길다면 해당 길이까지만 자르고 출력"""
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
        """ 최종후보가 k개인지 확인"""
        for done in self.sentences:
            if len(done) < self.k:
                return False
        return True

    def _forward_step(self, decoder_input, encoder_outputs):
        """ 각 셀마다 현재 상태에서 확률벡터를 구하고 출력"""
        decoder_input = decoder_input.to(self.device)
        output_size = decoder_input.size(1)
        embedded = self.embedding(decoder_input).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.use_attention:
            output, hidden, _ = self.rnn(inputs=embedded, pre_hidden=self.decoder_hidden, get_attention=True,
                                         attention=self.attention, encoder_outputs=encoder_outputs)  # decoder output
        else:
            output, hidden = self.rnn(inputs=embedded, pre_hidden=self.decoder_hidden)  # decoder output
        predicted_softmax = self.function(self.w(output.contiguous().view(-1, self.hidden_size)), dim=1).to(self.device)
        predicted_softmax = predicted_softmax.view(self.batch_size, output_size, -1)

        return predicted_softmax

    def _get_length_penalty(self, length, alpha=1.2, min_length=5):
        """ 확률은 0~1 사이이므로 길이가 길어질 수록 더 적아진다. 이를 보완하기 위해 길이에 따른 패널티를 부여하고 계산하며,
        일반적으로 alpha = 1.2, min_length = 5를 사용하며, 이는 수정가능하다."""
        return ((min_length + length) / (min_length + 1)) ** alpha

    def _replace_beam(self, child_ps, child_vs, done_ids, count):
        """ 만약에 end token이 나왔다면 그것을 최종후보에 등록시키고, k+1번째로 다시 전개시켜 k개로 맞춤"""
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