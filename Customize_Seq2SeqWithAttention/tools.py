# -*- coding:utf-8 -*-
import os
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as opt
from bleu import n_gram_precision
from torch.utils.data import DataLoader
from data_helper import create_or_get_voca, LSTMSeq2SeqDataset
from Customize_Seq2SeqWithAttention.model import Encoder, AttentionDecoder, Seq2SeqWithAttention
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager, rc


font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/NanumBarunGothic.ttf").get_name()
rc('font', family=font_name)
plt.rcParams.update({'font.size': 7})

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
        self.x_train_path = os.path.join(self.args.data_path, self.args.src_train_filename)  # train input 경로
        self.y_train_path = os.path.join(self.args.data_path, self.args.tar_train_filename)  # train target 경로
        self.x_val_path = os.path.join(self.args.data_path, self.args.src_val_filename)      # validation input 경로
        self.y_val_path = os.path.join(self.args.data_path, self.args.tar_val_filename)      # validation target 경로
        self.ko_voc, self.en_voc = self.get_voca()      # vocabulary
        self.train_loader = self.get_train_loader()     # train data loader
        self.val_loader = self.get_val_loader()         # validation data loader
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.en_voc['<pad>'])             # cross entropy
        self.writer = SummaryWriter()                   # tensorboard 기록
        self.train()                                    # train 실행

    def train(self):
        start = time.time()                             # 모델 시작 시간 기록
        encoder_parameter = self.encoder_parameter()    # encoder parameter
        decoder_parameter = self.decoder_parameter()    # decoder parameter

        encoder = Encoder(**encoder_parameter)          # encoder 초기화
        decoder = AttentionDecoder(**decoder_parameter) # decoder 초기화
        model = Seq2SeqWithAttention(encoder, decoder, self.args.sequence_size, self.args.get_attention) # model  초기화
        model = nn.DataParallel(model)                  # model을 여러개 GPU의 할당
        model.cuda()                                    # model의 모든 parameter를 GPU에 loading
        model.train()                                   # 모델을 훈련상태로

        # encoder, decoder optimizer 분리
        encoder_optimizer = opt.Adam(model.parameters(), lr=self.args.learning_rate)    # Encoder Adam Optimizer
        decoder_optimizer = opt.Adam(model.parameters(), lr=self.args.learning_rate)    # Decoder Adam Optimizer

        epoch_step = len(self.train_loader) + 1                                         # 전체 데이터 셋 / batch_size
        total_step = self.args.epochs * epoch_step                                      # 총 step 수
        train_ratios = cal_teacher_forcing_ratio(self.args.learning_method, total_step)     # train learning method
        val_ratios = cal_teacher_forcing_ratio('Scheduled_Sampling', int(total_step / 100)+1)  # val learning method

        step = 0

        attention = None

        for epoch in range(self.args.epochs):           # 매 epoch 마다
            for i, data in enumerate(self.train_loader, 0):     # train에서 data를 불러옴
                try:
                    src_input, tar_input, tar_output = data
                    if self.args.get_attention:     # attention model
                        output, attention = model(src_input, tar_input, teacher_forcing_rate=train_ratios[step])
                    else:   # seq2seq model
                        output = model(src_input, tar_input, teacher_forcing_rate=train_ratios[step])
                    # Get loss & accuracy & Perplexity
                    loss, accuracy, ppl = self.loss_accuracy(output, tar_output)

                    # Training Log
                    if step % self.args.train_step_print == 0:                              # train step마다
                        self.writer.add_scalar('train/loss', loss.item(), step)             # save loss to tensorboard
                        self.writer.add_scalar('train/accuracy', accuracy.item(), step)     # save accuracy to tb
                        self.writer.add_scalar('train/PPL', ppl, step)                      # save Perplexity to tb

                        print('[Train] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                              '=>  loss : {5:10f}  accuracy : {6:12f}  PPL : {7:6f}  tf_rate : {8:6f}'
                              .format(epoch, i, epoch_step, step, total_step, loss.item(), accuracy.item(), ppl,
                                      train_ratios[step]))

                    # Validation Log
                    if step % self.args.val_step_print == 0:        # validation step마다
                        with torch.no_grad():                       # validation은 학습되지 않음
                            model.eval()                            # 모델을 평가상태로
                            if step >= self.args.val_step_print:    # validation step은 100번마다 1번이므로 이에 따라 설정
                                steps = int(step / self.args.val_step_print)
                            else:
                                steps = step
                            val_loss, val_accuracy, val_ppl, val_bleu = self.val(model,
                                                                                 teacher_forcing_rate=val_ratios[steps])
                            self.writer.add_scalar('val/loss', val_loss, step)          # save loss to tb
                            self.writer.add_scalar('val/accuracy', val_accuracy, step)  # save accuracy to tb
                            self.writer.add_scalar('val/PPL', val_ppl, step)            # save PPl to tb
                            self.writer.add_scalar('val/BLEU', val_bleu, step)          # save BLEU to tb
                            print('[Val] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                                  '=>  loss : {5:10f}  accuracy : {6:12f}   PPL : {7:10f}  tf_rate : {8:6f}'
                                  .format(epoch, i, epoch_step, step, total_step, val_loss, val_accuracy, val_ppl,
                                          val_ratios[steps]))
                            model.train()           # 모델을 훈련상태로

                    # Save Model Point
                    if step % self.args.step_save == 0:         # save step마다
                        print("time :", time.time() - start)    # 걸린시간 출력
                        if self.args.get_attention:
                            self.plot_attention(step, src_input, tar_input, attention)      # attention 그리기
                        self.model_save(model=model, encoder_optimizer=encoder_optimizer,
                                        decoder_optimizer=decoder_optimizer, epoch=epoch, step=step)

                    encoder_optimizer.zero_grad()   # encoder optimizer 모든 변화도 0
                    decoder_optimizer.zero_grad()   # decoder optimizer 모든 변화도 0
                    loss.backward()                 # 역전파 단계
                    encoder_optimizer.step()        # encoder 매개변수 갱신
                    decoder_optimizer.step()        # decoder 매개변수 갱신
                    step += 1

                # If KeyBoard Interrupt Save Model
                except KeyboardInterrupt:
                    self.model_save(model=model, encoder_optimizer=encoder_optimizer,
                                    decoder_optimizer=decoder_optimizer, epoch=epoch, step=step)

    def get_voca(self):
        try:    # vocabulary 불러오기
            ko_voc, en_voc = create_or_get_voca(save_path=self.args.dictionary_path)
        except OSError:     # 경로 error 발생 시 각각의 경로를 입력해서 가지고 오기
            ko_voc, en_voc = create_or_get_voca(save_path=self.args.dictionary_path,
                                                ko_corpus_path=self.x_train_path,
                                                di_corpus_path=self.y_train_path)
        return ko_voc, en_voc

    def get_train_loader(self):
        # 재현을 위해 랜덤시드 고정
        # seed_val = 42
        # torch.manual_seed(seed_val)
        # path를 불러와서 train_loader를 만드는 함수
        train_dataset = LSTMSeq2SeqDataset(self.x_train_path, self.y_train_path, self.ko_voc, self.en_voc,
                                           self.args.sequence_size)
        point_sampler = torch.utils.data.RandomSampler(train_dataset)   # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=point_sampler)
        return train_loader

    def get_val_loader(self):
        # 재현을 위해 랜덤시드 고정
        # seed_val = 42
        # torch.manual_seed(seed_val)
        # path를 불러와서 train_loader를 만드는 함수
        val_dataset = LSTMSeq2SeqDataset(self.x_val_path, self.y_val_path, self.ko_voc, self.en_voc,
                                         self.args.sequence_size)
        point_sampler = torch.utils.data.RandomSampler(val_dataset)     # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, sampler=point_sampler)
        return val_loader

    # Encoder Parameter
    def encoder_parameter(self):
        param = {
            'embedding_size': self.args.embedding_size,
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

    # Decoder Parameter
    def decoder_parameter(self):
        param = {
            'embedding_size': self.args.embedding_size,
            'embedding_dim': self.args.embedding_dim,
            'pad_id': self.en_voc['<pad>'],
            'rnn_dim': self.args.decoder_rnn_dim,
            'rnn_bias': True,
            'n_layers': self.args.decoder_n_layers,
            'embedding_dropout': self.args.decoder_embedding_dropout,
            'rnn_dropout': self.args.decoder_rnn_dropout,
            'dropout': self.args.decoder_dropout,
            'residual_used': self.args.decoder_residual_used,
            'attention_score_func': self.args.attention_score
        }
        return param

    # calculate loss, accuracy, Perplexity
    def loss_accuracy(self, out, tar):
        # out => [batch_size, sequence_len, vocab_size]
        # tar => [batch_size, sequence_len]
        out = out.view(-1, out.size(-1))
        tar = tar.view(-1).to(device)

        # out => [batch_size * sequence_len, vocab_size]
        # tar => [batch_size * sequence_len]
        loss = self.criterion(out, tar)     # calculate loss with CrossEntropy
        ppl = math.exp(loss.item())         # perplexity = exponential(loss)

        indices = out.max(-1)[1]         # 배열의 최대 값이 들어 있는 index 리턴

        invalid_targets = tar.eq(self.en_voc['<pad>'])  # tar 에 있는 index 중 pad index가 있으면 True, 없으면 False
        equal = indices.eq(tar)                         # target이랑 indices 비교
        total = 0
        for i in invalid_targets:
            if i == 0:
                total += 1
        accuracy = torch.div(equal.masked_fill_(invalid_targets, 0).long().sum().to(dtype=torch.float32), total)
        return loss, accuracy, ppl

    def val(self, model, teacher_forcing_rate):
        total_loss = 0
        total_accuracy = 0
        total_ppl = 0
        with torch.no_grad():   # 기록하지 않음
            count = 0
            for data in self.val_loader:
                src_input, tar_input, tar_output = data
                output = model(src_input, tar_input, teacher_forcing_rate=0)

                if isinstance(output, tuple):   # attention이 같이 출력되는 경우 output만
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
            output_sentence = self.tensor2sentence_en(indices)
            target_sentence = self.tensor2sentence_en(b)
            bleu_score = n_gram_precision(output_sentence[0], target_sentence[0])
            print("-------test-------")
            print("Korean: ", self.tensor2sentence_ko(a))           # input 출력
            print("Predicted : ", output_sentence)                  # output 출력
            print("Target :", target_sentence)                      # target 출력
            print('BLEU Score : ', bleu_score)
            avg_loss = total_loss / count                           # 평균 loss
            avg_accuracy = total_accuracy / count                   # 평균 accuracy
            avg_ppl = total_ppl / count                             # 평균 Perplexity
            return avg_loss, avg_accuracy, avg_ppl, bleu_score

    def model_save(self, model, encoder_optimizer, decoder_optimizer, epoch, step):
        model_name = '{0:06d}_model_1.pth'.format(step)                 # 모델파일 이름
        model_path = os.path.join(self.args.model_path, model_name)     # 모델저장 경로
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
            if word == '</s>':      # End token 나오면 stop
                break
            translation_sentence.append(word)
        translation_sentence = ''.join(translation_sentence).replace('▁', ' ').strip()  # sentencepiece 에 _ 제거
        result.append(translation_sentence)
        return result

    def tensor2sentence_ko(self, indices: torch.Tensor) -> list:
        result = []
        translation_sentence = []
        for idx in indices:
            word = self.ko_voc.IdToPiece(idx)
            if word == '<pad>':                 # padding 나오면 끝
                break
            translation_sentence.append(word)
        translation_sentence = ''.join(translation_sentence).replace('▁', ' ').strip()  # sentencepiece 에 _ 제거
        result.append(translation_sentence)
        return result

    def plot_attention(self, step, src_input, trg_input, attention):    # Attention 그리기 함수
        filename = '{0:06d}_step'.format(step)
        filepath = os.path.join(self.args.img_path, filename)
        try:
            os.mkdir(filepath)
        except FileExistsError:
            pass

        def replace_pad(words):
            return [word if word != '<pad>' else '' for word in words]

        with torch.no_grad():
            src_input = src_input.to('cpu')
            trg_input = trg_input.to('cpu')
            attention = attention.to('cpu')

            sample = [i for i in range(src_input.shape[0] - 1)]
            sample = random.sample(sample, self.args.plot_count)

            for num, i in enumerate(sample):
                src, trg = src_input[i], trg_input[i]
                src_word = replace_pad([self.ko_voc.IdToPiece(word.item()) for word in src])
                trg_word = replace_pad([self.en_voc.IdToPiece(word.item()) for word in trg])

                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.matshow(attention[i].data, cmap='bone')
                fig.colorbar(cax)

                ax.set_xticklabels(trg_word, rotation=90)
                ax.set_yticklabels(src_word)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                fig.savefig(fname=os.path.join(filepath, 'attention-{}.png'.format(num)))
            plt.close('all')


class Translation(object):  # Usage
    def __init__(self, checkpoint, dictionary_path, x_path=None, y_path=None, beam_search=False, k=1,
                 get_attention=False):
        self.checkpoint = torch.load(checkpoint)    # model load
        self.seq_len = self.checkpoint['seq_len']   # sequence 길이
        self.batch_size = 100                       # batch size
        self.x_path = x_path                        # encoder input path
        self.y_path = y_path                        # decoder input path
        self.beam_search = beam_search              # beam search 사용여부
        self.k = k                                  # beam search parameter k
        self.ko_voc, self.en_voc = create_or_get_voca(save_path=dictionary_path)    # vocabulary
        self.get_attention = get_attention          # attention 사용 여부
        self.model = self.model_load()              # model

    def model_load(self):
        encoder = Encoder(**self.checkpoint['encoder_parameter'])               # encoder
        decoder = AttentionDecoder(**self.checkpoint['decoder_parameter'])      # decoder
        model = Seq2SeqWithAttention(encoder, decoder, self.seq_len, get_attention=self.get_attention,
                                     beam_search=self.beam_search, k=self.k)
        model = nn.DataParallel(model)                                          # GPU: Data Paraller
        model.load_state_dict(self.checkpoint['model_state_dict'])              # model parameter
        model.eval()                                                            # evaluate mode
        return model

    def src_input(self, sentence):
        idx_list = self.ko_voc.EncodeAsIds(sentence)                            # sentence -> ids
        idx_list = self.padding(idx_list, self.ko_voc['<pad>'])                 # padding
        return torch.tensor([idx_list]).to(device)

    def tar_input(self):
        idx_list = [self.en_voc['<s>']]                                         # add start token
        idx_list = self.padding(idx_list, self.en_voc['<pad>'])                 # padding
        return torch.tensor([idx_list]).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)
        if length < self.seq_len:                                               # 문장길이가 sequence_len보다 짧으면
            idx_list = idx_list + [padding_id for _ in range(self.seq_len - len(idx_list))]  # 남은 길이만큼 padding추가
        else:                                                                   # 문장길이가 sequence_len보다 길면
            idx_list = idx_list[:self.seq_len]
        return idx_list

    def tensor2sentence(self, indices: torch.Tensor) -> list:
        translation_sentence = []
        for idx in indices:
            word = self.en_voc.IdToPiece(idx)
            if word == '</s>':
                break
            translation_sentence.append(word)
        translation_sentence = ''.join(translation_sentence).replace('▁', ' ').strip()
        return translation_sentence

    def get_test_loader(self):
        with open(self.x_path, 'r', encoding='utf-8') as f:
            src_list = []
            for line in f:
                src_list.append(line)

        with open(self.y_path, 'r', encoding='utf-8') as f:
            tar_list = []
            for line in f:
                tar_list.append(line)
        return src_list[:self.batch_size], tar_list[:self.batch_size]

    def transform(self, sentence: str) -> (str, torch.Tensor):
        src_input = self.src_input(sentence)
        tar_input = self.tar_input()
        output = self.model(src_input, tar_input, teacher_forcing_rate=0)
        if isinstance(output, tuple):  # attention이 같이 출력되는 경우 output만
            output = output[0]
        _, indices = output.view(-1, output.size(-1)).max(-1)

        pred = self.tensor2sentence(indices.tolist())

        print('Korean: ' + sentence)
        print('Predict: ' + pred)

    def batch_transform(self):
        src_list, tar_list = self.get_test_loader()

        if len(src_list) > self.batch_size:
            raise ValueError('You must sentence size less than {}'.format(self.batch_size))

        src_inputs = torch.stack([self.src_input(sentence) for sentence in src_list]).squeeze(dim=1)
        tar_inputs = torch.stack([self.tar_input() for _ in src_list]).squeeze(dim=1)

        output = self.model(src_inputs, tar_inputs, teacher_forcing_rate=0)
        if isinstance(output, tuple):  # attention이 같이 출력되는 경우 output만
            output = output[0]
        _, indices = output.view(-1, output.size(-1)).max(-1)
        result = []

        for i in range(0, len(indices), self.seq_len):
            end = i + self.seq_len - 1
            indices_i = indices[i:end].tolist()
            result.append(self.tensor2sentence(indices_i))

        for src, tar, pred in zip(src_list, tar_list, result):
            print('Korean: ' + src, end='')
            print('English: ' + tar, end='')
            print('Predict: ' + pred)
            print('-------------------------')

