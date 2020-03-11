import random
import torch
import torch.nn as nn


class StackLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        for i in range(n_layers):  # LSTM 층 쌓기
            self.layers.append(
                nn.LSTMCell(input_size, hidden_size, bias=bias)
            )
            input_size = hidden_size

    def forward(self, inputs, hidden):
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
        return inputs, next_hidden


class Recurrent(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, inputs, hidden=None, context=None):
        hidden_size = self.cell.hidden_size
        batch_size = inputs.size()[0]

        if hidden is None:
            n_layers = self.cell.n_layers
            zero = inputs.data.new(1).zero_()
            # hidden 초기화
            h0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            # cell 초기화
            c0 = zero.view(1, 1, 1).expand(n_layers, batch_size, hidden_size)
            hidden = (h0, c0)

        outputs = []
        inputs_time = inputs.split(1, dim=1)

        for input_t in inputs_time:
            input_t = input_t.squeeze(1)
            output_t, hidden = self.cell(input_t, hidden)
            outputs += [output_t]

        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden


class Encoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)

        # rnn cell
        cell = StackLSTMCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim, n_layers=n_layers,
                             bias=rnn_bias)

        self.rnn = Recurrent(cell)

    def forward(self, enc_input):
        embedded = self.embedding(enc_input)
        # embedded => [batch_size, sequence_len, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded)
        # output => [batch_size, sequence_len, rnn_dim]
        # hidden => [n_layer, batch_size, rnn_dim]
        # cell => [n_layer, batch_size, rnn_dim]
        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, embedding_size, embedding_dim, rnn_dim, rnn_bias, pad_id, n_layers=1):
        super().__init__()
        self.vocab_size = embedding_size
        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=pad_id)
        cell = StackLSTMCell(input_size=self.embedding.embedding_dim, hidden_size=rnn_dim, n_layers=n_layers,
                             bias=rnn_bias)
        self.rnn = Recurrent(cell)
        self.classifier = nn.Linear(rnn_dim, embedding_size)

    def forward(self, context, dec_input, hidden, scheduled_sampling=False):
        embedded = self.embedding(dec_input)
        output, hidden = self.rnn(inputs=embedded, hidden=hidden, context=context)
        # output => [batch_size, sequence_size, rnn_dim]
        output = self.classifier(output)
        # output => [batch_size, sequence_size, embedding_size]
        if scheduled_sampling:
            return output, hidden
        else:
            output = nn.functional.log_softmax(output, dim=1)
        # output => [batch_size, sequence_size, embedding_size]
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, seq_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len = seq_len

    def forward(self, enc_input, dec_input, teacher_forcing_rate=0.5):
        context, hidden = self.encoder(enc_input)
        seed_val = 42
        random.seed(seed_val)
        # teacher forcing ratio check
        if teacher_forcing_rate == 1.0:  # 교사강요 무조건 적용  => 답을 그대로 다음 input에 넣음
            output, _ = self.decoder(context=context, dec_input=dec_input, hidden=hidden)
            return output
        else:
            scheduled_sampling = True
            outputs = []
            dec_input_i = dec_input[:, 0].unsqueeze(dim=1)

            for i in range(1, self.seq_len+1):
                output, hidden = self.decoder(context=context, dec_input=dec_input_i, hidden=hidden,
                                              scheduled_sampling=scheduled_sampling)
                _, indices = output.max(dim=2)

                output = output.squeeze(dim=1)
                outputs.append(output)

                if i != self.seq_len:
                    dec_input_i = dec_input[:, i].unsqueeze(dim=1) if random.random() < teacher_forcing_rate \
                        else indices

            outputs = torch.stack(outputs, dim=1)
            outputs = nn.functional.log_softmax(outputs, dim=1)
            return outputs
