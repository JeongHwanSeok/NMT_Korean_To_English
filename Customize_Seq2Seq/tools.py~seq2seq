# -*- coding:utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from data_helper import create_or_get_voca, LSTMSeq2SeqDataset
from Customize_Seq2Seq2.model import Encoder, Decoder, Seq2Seq
from tensorboardX import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def cal_teacher_forcing_ratio(learning_method, total_step):
    if learning_method == 'Teacher_Forcing':
        teacher_forcing_ratios = [1.0 for _ in range(total_step)]  # 교사강요
    elif learning_method == 'Scheduled_Sampling':
        import numpy as np
        teacher_forcing_ratios = np.linspace(0.0, 1.0, num=total_step)[::-1]  # 스케줄 샘플링
        # np.linspace : 시작점과 끝점을 균일하게 toptal_step수 만큼 나눈 점을 생성
    else:
        raise NotImplementedError('learning method must choice [Teacher_Forcing, Scheduled_Sampling]')
    return teacher_forcing_ratios


class Trainer(object):  # Train
    def __init__(self, args):
        self.args = args
        self.x_train_path = os.path.join(self.args.data_path, self.args.src_train_filename)
        self.y_train_path = os.path.join(self.args.data_path, self.args.tar_train_filename)
        self.x_val_path = os.path.join(self.args.data_path, self.args.src_val_filename)
        self.y_val_path = os.path.join(self.args.data_path, self.args.tar_val_filename)
        self.ko_voc, self.en_voc = self.get_voca()
        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_val_loader()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.en_voc['<pad>'])
        self.writer = SummaryWriter()
        self.train()

    def train(self):
        encoder_parameter = self.encoder_parameter()
        decoder_parameter = self.decoder_parameter()

        encoder = Encoder(**encoder_parameter)
        decoder = Decoder(**decoder_parameter)
        model = Seq2Seq(encoder, decoder, self.args.sequence_size)
        model = nn.DataParallel(model)
        model.cuda()
        model.train()

        encoder_optimizer = opt.Adam(model.parameters(), lr=self.args.learning_rate)
        decoder_optimizer = opt.Adam(model.parameters(), lr=self.args.learning_rate)

        epoch_step = len(self.train_loader) + 1
        total_step = self.args.epochs * epoch_step
        train_ratios = cal_teacher_forcing_ratio(self.args.learning_method, total_step)
        val_ratios = cal_teacher_forcing_ratio('Scheduled_Sampling', int(total_step / 100)+1)

        step = 0

        for epoch in range(self.args.epochs):
            for i, data in enumerate(self.train_loader, 0):
                try:
                    src_input, tar_input, tar_output = data
                    output = model(src_input, tar_input, teacher_forcing_rate=train_ratios[i])
                    # Get loss & accuracy
                    loss, accuracy, ppl = self.loss_accuracy(output, tar_output)

                    # Training Log
                    if step % self.args.train_step_print == 0:
                        self.writer.add_scalar('train/loss', loss.item(), step)
                        self.writer.add_scalar('train/accuracy', accuracy.item(), step)
                        self.writer.add_scalar('train/PPL', ppl, step)

                        print('[Train] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                              '=>  loss : {5:10f}  accuracy : {6:12f}  PPL : {7:6f}'
                              .format(epoch, i, epoch_step, step, total_step, loss.item(), accuracy.item(), ppl))

                    # Validation Log
                    if step % self.args.val_step_print == 0:
                        with torch.no_grad():
                            model.eval()
                            if step >= 100:
                                steps = int(step / 100)
                            else:
                                steps = step
                            val_loss, val_accuracy, val_ppl = self.val(model,
                                                                       teacher_forcing_rate=val_ratios[steps])
                            self.writer.add_scalar('val/loss', val_loss, step)
                            self.writer.add_scalar('val/accuracy', val_accuracy, step)
                            self.writer.add_scalar('val/PPL', val_ppl, step)

                            print('[Val] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                                  '=>  loss : {5:10f}  accuracy : {6:12f}   PPL : {7:10f}'
                                  .format(epoch, i, epoch_step, step, total_step, val_loss, val_accuracy, val_ppl))
                            model.train()

                    # Save Model Point
                    if step % self.args.step_save == 0:
                        self.model_save(model=model, encoder_optimizer=encoder_optimizer,
                                        decoder_optimizer=decoder_optimizer, epoch=epoch, step=step)

                    # optimizer
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    loss.backward()
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    step += 1

                # If KeyBoard Interrupt Save Model
                except KeyboardInterrupt:
                    self.model_save(model=model, encoder_optimizer=encoder_optimizer,
                                    decoder_optimizer=decoder_optimizer, epoch=epoch, step=step)

    def get_voca(self):
        try:
            ko_voc, en_voc = create_or_get_voca(save_path=self.args.dictionary_path)
        except OSError:
            ko_voc, en_voc = create_or_get_voca(save_path=self.args.dictionary_path,
                                                ko_corpus_path=self.x_train_path,
                                                en_corpus_path=self.y_train_path)
        return ko_voc, en_voc

    def get_train_loader(self):
        # 재현을 위해 랜덤시드 고정
        seed_val = 42
        torch.manual_seed(seed_val)
        # path를 불러와서 train_loader를 만드는 함수
        train_dataset = LSTMSeq2SeqDataset(self.x_train_path, self.y_train_path, self.ko_voc, self.en_voc,
                                           self.args.sequence_size)
        point_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=point_sampler)
        return train_loader

    def get_val_loader(self):
        # 재현을 위해 랜덤시드 고정
        seed_val = 42
        torch.manual_seed(seed_val)
        # path를 불러와서 train_loader를 만드는 함수
        val_dataset = LSTMSeq2SeqDataset(self.x_val_path, self.y_val_path, self.ko_voc, self.en_voc,
                                         self.args.sequence_size)
        point_sampler = torch.utils.data.RandomSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, sampler=point_sampler)
        return val_loader

    def encoder_parameter(self):
        param = {
            'embedding_size': 5000,
            'embedding_dim': self.args.embedding_dim,
            'pad_id': self.ko_voc['<pad>'],
            'rnn_dim': self.args.encoder_rnn_dim,
            'rnn_bias': True,
            'n_layers': self.args.encoder_n_layers,
            'embedding_dropout': self.args.encoder_embedding_dropout,
            'rnn_dropout': self.args.encoder_rnn_dropout,
            'dropout': self.args.encoder_dropout,
            'residual_used': self.args.encoder_residual_used,
            'bidirectional': self.args.encoder_bidirectional_used,
            'encoder_output_transformer': self.args.encoder_output_transformer,
            'encoder_output_transformer_bias': self.args.encoder_output_transformer_bias,
            'encoder_hidden_transformer': self.args.encoder_hidden_transformer,
            'encoder_hidden_transformer_bias': self.args.encoder_hidden_transformer_bias
        }
        return param

    def decoder_parameter(self):
        param = {
            'embedding_size': 5000,
            'embedding_dim': self.args.embedding_dim,
            'pad_id': self.en_voc['<pad>'],
            'rnn_dim': self.args.decoder_rnn_dim,
            'rnn_bias': True,
            'n_layers': self.args.decoder_n_layers,
            'embedding_dropout': self.args.decoder_embedding_dropout,
            'rnn_dropout': self.args.decoder_rnn_dropout,
            'dropout': self.args.decoder_dropout,
            'residual_used': self.args.decoder_residual_used
        }
        return param

    def loss_accuracy(self, out, tar):
        # out => [embedding_size, sequence_len, vocab_size]
        # tar => [embedding_size, sequence_len]
        out = out.view(-1, out.size(-1))
        tar = tar.view(-1).to(device)
        # out => [embedding_size * sequence_len, vocab_size]
        # tar => [embedding_size * sequence_len]
        loss = self.criterion(out, tar)
        ppl = math.exp(loss.item())

        _, indices = out.max(-1)
        invalid_targets = tar.eq(self.en_voc['<pad>'])
        equal = indices.eq(tar)
        total = 1
        for i in equal.size():
            total *= i
        accuracy = torch.div(equal.masked_fill_(invalid_targets, 0).long().sum().to(dtype=torch.float32), total)
        return loss, accuracy, ppl

    def val(self, model, teacher_forcing_rate):
        total_loss = 0
        total_accuracy = 0
        total_ppl = 0
        with torch.no_grad():
            count = 0
            for data in self.val_loader:
                src_input, tar_input, tar_output = data
                output = model(src_input, tar_input, teacher_forcing_rate=teacher_forcing_rate)
                if isinstance(output, tuple):
                    output = output[0]
                loss, accuracy, ppl = self.loss_accuracy(output, tar_output)
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_ppl += ppl
                count += 1
            _, indices = output.view(-1, output.size(-1)).max(-1)
            indices = indices[:self.args.sequence_size].tolist()
            a = src_input[0].tolist()
            b = tar_output[0].tolist()
            print(self.tensor2sentence_ko(a))
            print(self.tensor2sentence_en(indices))
            print(self.tensor2sentence_en(b))
            avg_loss = total_loss / count
            avg_accuracy = total_accuracy / count
            avg_ppl = total_ppl / count
            return avg_loss, avg_accuracy, avg_ppl

    def model_save(self, model, encoder_optimizer, decoder_optimizer, epoch, step):
        model_name = '{0:06d}_model_1.pth'.format(step)
        model_path = os.path.join(self.args.model_path, model_name)
        torch.save({
            'epoch': epoch,
            'steps': step,
            'seq_len': self.args.sequence_size,
            'encoder_parameter': self.encoder_parameter(),
            'decoder_parameter': self.decoder_parameter(),
            'model_state_dict': model.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict()

        }, model_path)

    def tensor2sentence_en(self, indices: torch.Tensor) -> list:
        result = []
        translation_sentence = []
        for idx in indices:
            word = self.en_voc.IdToPiece(idx)
            if word == '</s>':
                break
            translation_sentence.append(word)
        translation_sentence = ''.join(translation_sentence).replace('▁', ' ').strip()
        result.append(translation_sentence)
        return result

    def tensor2sentence_ko(self, indices: torch.Tensor) -> list:
        result = []
        translation_sentence = []
        for idx in indices:
            word = self.ko_voc.IdToPiece(idx)
            if word == '<pad>':
                break
            translation_sentence.append(word)
        translation_sentence = ''.join(translation_sentence).replace('▁', ' ').strip()
        result.append(translation_sentence)
        return result


class Translation(object):  # Usage
    def __init__(self, checkpoint, dictionary_path, x_path=None, y_path=None, beam_search=False, k=1):
        self.checkpoint = torch.load(checkpoint)
        self.seq_len = self.checkpoint['seq_len']
        self.batch_size = 100
        self.x_path = x_path
        self.y_path = y_path
        self.beam_search = beam_search
        self.k = k
        self.ko_voc, self.en_voc = create_or_get_voca(save_path=dictionary_path)
        self.model = self.model_load()

    def model_load(self):
        encoder = Encoder(**self.checkpoint['encoder_parameter'])
        decoder = Decoder(**self.checkpoint['decoder_parameter'])
        model = Seq2Seq(encoder, decoder, self.seq_len, beam_search=self.beam_search, k=self.k)
        model = nn.DataParallel(model)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        return model

    def src_input(self, sentence):
        idx_list = self.ko_voc.EncodeAsIds(sentence)
        idx_list = self.padding(idx_list, self.ko_voc['<pad>'])
        return torch.tensor([idx_list]).to(device)

    def tar_input(self):
        idx_list = [self.en_voc['<s>']]
        return torch.tensor([idx_list]).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)
        if length < self.seq_len:
            idx_list = idx_list + [padding_id for _ in range(self.seq_len - len(idx_list))]
        else:
            idx_list = idx_list[:self.seq_len]
        return idx_list

    def tensor2sentence(self, indices: torch.Tensor) -> list:
        result = []

        for idx_list in indices:
            translation_sentence = []
            for idx in idx_list:
                word = self.en_voc.IdToPiece(idx.item())
                if word == '</s>':
                    break
                translation_sentence.append(word)
            translation_sentence = ''.join(translation_sentence).replace('▁', ' ').strip()
            result.append(translation_sentence)
        return result

    def get_test_loader(self):
        with open(self.x_path, 'r', encoding='utf-8') as f:
            src_list = []
            for line in f:
                src_list.append(line)

        with open(self.y_path, 'r', encoding='utf-8') as f:
            tar_list = []
            for line in f:
                tar_list.append(line)

        return src_list[:10], tar_list[:10]

    def transform(self, sentence: str) -> (str, torch.Tensor):
        src_input = self.src_input(sentence)
        tar_input = self.tar_input()
        output = self.model(src_input, tar_input)
        result = self.tensor2sentence(output)[0]
        return result

    def batch_transform(self):
        src_list, tar_list = self.get_test_loader()

        if len(src_list) > self.batch_size:
            raise ValueError('You must sentence size less than {}'.format(self.batch_size))

        src_inputs = torch.stack([self.src_input(sentence) for sentence in src_list]).squeeze(dim=1)
        tar_inputs = torch.stack([self.tar_input() for _ in src_list]).squeeze(dim=1)

        output = self.model(src_inputs, tar_inputs)

        result = self.tensor2sentence(output)
        for src, tar, pred in zip(src_list, tar_list, result):
            print('Korean: ', src)
            print('English: ', tar)
            print('Predict: ', pred)
        return result
