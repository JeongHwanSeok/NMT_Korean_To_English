# -*- coding:utf-8 -*-
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as opt
from bleu import n_gram_precision
from torch.utils.data import DataLoader
from data_helper import create_or_get_voca, LSTMSeq2SeqDataset
from Transformer.model import Encoder, Decoder, Transformer, greedy_decoder, Beam
from Transformer.utils import NoamOpt, CrossEntropyLoss, EarlyStopping
from tensorboardX import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class Trainer(object):  # Train
    def __init__(self, args):
        self.args = args
        self.x_train_path = os.path.join(self.args.data_path, self.args.src_train_filename)  # train input 경로
        self.y_train_path = os.path.join(self.args.data_path, self.args.tar_train_filename)  # train target 경로
        self.x_val_path = os.path.join(self.args.data_path, self.args.src_val_filename)      # validation input 경로
        self.y_val_path = os.path.join(self.args.data_path, self.args.tar_val_filename)      # validation target 경로
        self.ko_voc, self.di_voc = self.get_voca()      # vocabulary
        self.train_loader = self.get_train_loader()     # train data loader
        self.val_loader = self.get_val_loader()         # validation data loader           # cross entropy
        self.criterion = CrossEntropyLoss(ignore_index=self.di_voc['<pad>'], smooth_eps=args.label_smoothing)
        self.writer = SummaryWriter()                   # tensorboard 기록
        self.early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)
        self.train()                                    # train 실행

    def train(self):
        start = time.time()                             # 모델 시작 시간 기록
        encoder_parameter = self.encoder_parameter()    # encoder parameter
        decoder_parameter = self.decoder_parameter()    # decoder parameter

        encoder = Encoder(**encoder_parameter)          # Encoder 초기화
        decoder = Decoder(**decoder_parameter)          # Decoder 초기화

        model = Transformer(encoder, decoder)
        model = nn.DataParallel(model)                  # model을 여러개 GPU의 할당
        model.cuda()                                    # model의 모든 parameter를 GPU에 loading
        model.train()                                   # 모델을 훈련상태로

        print(f'The model has {count_parameters(model):,} trainable parameters')
        model.apply(initialize_weights)

        # encoder, decoder optimizer 분리
        # optimizer = opt.Adam(model.parameters(), lr=self.args.learning_rate)    # Encoder Adam Optimizer
        optimizer = NoamOpt(self.args.encoder_hidden_dim, 1, 4000,
                            opt.Adam(model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.98),
                                     weight_decay=0.0001))
        epoch_step = len(self.train_loader) + 1                                         # 전체 데이터 셋 / batch_size
        total_step = self.args.epochs * epoch_step                                      # 총 step 수
        step = 0

        for epoch in range(self.args.epochs):           # 매 epoch 마다
            for i, data in enumerate(self.train_loader, 0):     # train에서 data를 불러옴
                try:
                    optimizer.optimizer.zero_grad()  # encoder optimizer 모든 변화도 0
                    src_input, tar_input, tar_output = data
                    output, attention = model(src_input, tar_input)
                    loss, accuracy, ppl = self.loss_accuracy(output, tar_output)

                    # Training Log
                    if step % self.args.train_step_print == 0:                              # train step마다
                        self.writer.add_scalar('train/loss', loss.item(), step)             # save loss to tensorboard
                        self.writer.add_scalar('train/accuracy', accuracy.item(), step)     # save accuracy to tb
                        self.writer.add_scalar('train/PPL', ppl, step)                      # save Perplexity to tb

                        print('[Train] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                              '=>  loss : {5:10f}  accuracy : {6:12f}  PPL : {7:6f}'
                              .format(epoch, i, epoch_step, step, total_step, loss.item(), accuracy.item(), ppl))

                    # Validation Log
                    if step % self.args.val_step_print == 0:        # validation step마다
                        with torch.no_grad():                       # validation은 학습되지 않음
                            model.eval()                            # 모델을 평가상태로
                            val_loss, val_accuracy, val_ppl, val_bleu = self.val(model)
                            self.writer.add_scalar('val/loss', val_loss, step)          # save loss to tb
                            self.writer.add_scalar('val/accuracy', val_accuracy, step)  # save accuracy to tb
                            self.writer.add_scalar('val/PPL', val_ppl, step)            # save PPl to tb
                            self.writer.add_scalar('val/BLEU', val_bleu, step)          # save BLEU to tb
                            print('[Val] epoch : {0:2d}  iter: {1:4d}/{2:4d}  step : {3:6d}/{4:6d}  '
                                  '=>  loss : {5:10f}  accuracy : {6:12f}   PPL : {7:10f}'
                                  .format(epoch, i, epoch_step, step, total_step, val_loss, val_accuracy, val_ppl))
                            self.early_stopping(val_loss, model, step, self.encoder_parameter(),
                                                self.decoder_parameter(), self.args.sequence_size)

                            model.train()           # 모델을 훈련상태로

                    # Save Model Point
                    if step % self.args.step_save == 0:         # save step마다
                        print("time :", time.time() - start)    # 걸린시간 출력
                        self.model_save(model=model,  epoch=epoch, step=step)
                    if self.early_stopping.early_stop:
                        print("Early Stopping")
                        raise KeyboardInterrupt
                    loss.backward()                 # 역전파 단계
                    optimizer.step()        # encoder 매개변수 갱신
                    step += 1

                # If KeyBoard Interrupt Save Model
                except KeyboardInterrupt:
                    self.model_save(model=model,  epoch=epoch, step=step)

    def get_voca(self):
        try:    # vocabulary 불러오기
            ko_voc, di_voc = create_or_get_voca(save_path=self.args.dictionary_path)
        except OSError:     # 경로 error 발생 시 각각의 경로를 입력해서 가지고 오기
            ko_voc, di_voc = create_or_get_voca(save_path=self.args.dictionary_path,
                                                ko_corpus_path=self.x_train_path,
                                                di_corpus_path=self.y_train_path)
        return ko_voc, di_voc

    def get_train_loader(self):
        # 재현을 위해 랜덤시드 고정
        # seed_val = 42
        # torch.manual_seed(seed_val)
        # path를 불러와서 train_loader를 만드는 함수
        train_dataset = LSTMSeq2SeqDataset(self.x_train_path, self.y_train_path, self.ko_voc, self.di_voc,
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
        val_dataset = LSTMSeq2SeqDataset(self.x_val_path, self.y_val_path, self.ko_voc, self.di_voc,
                                         self.args.sequence_size)
        point_sampler = torch.utils.data.RandomSampler(val_dataset)     # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, sampler=point_sampler)
        return val_loader

    # Encoder Parameter
    def encoder_parameter(self):
        param = {
            'input_dim': self.args.encoder_vocab_size,
            'hid_dim': self.args.encoder_hidden_dim,
            'n_layers': self.args.encoder_layers,
            'n_heads': self.args.encoder_heads,
            'head_dim': self.args.encoder_head_dim,
            'pf_dim': self.args.encoder_pf_dim,
            'dropout': self.args.encoder_dropout,
            'max_length': self.args.sequence_size,
            'padding_id': self.ko_voc['<pad>']
        }
        return param

    # Decoder Parameter
    def decoder_parameter(self):
        param = {
            'input_dim': self.args.decoder_vocab_size,
            'hid_dim': self.args.decoder_hidden_dim,
            'n_layers': self.args.decoder_layers,
            'n_heads': self.args.decoder_heads,
            'head_dim': self.args.decoder_head_dim,
            'pf_dim': self.args.decoder_pf_dim,
            'dropout': self.args.decoder_dropout,
            'max_length': self.args.sequence_size,
            'padding_id': self.di_voc['<pad>']
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
        invalid_targets = tar.eq(self.di_voc['<pad>'])  # tar 에 있는 index 중 pad index가 있으면 True, 없으면 False
        equal = indices.eq(tar)                         # target이랑 indices 비교
        total = 0
        for i in invalid_targets:
            if i == 0:
                total += 1
        accuracy = torch.div(equal.masked_fill_(invalid_targets, 0).long().sum().to(dtype=torch.float32), total)
        return loss, accuracy, ppl

    def val(self, model):
        total_loss = 0
        total_accuracy = 0
        total_ppl = 0
        with torch.no_grad():   # 기록하지 않음
            count = 0
            for i, data in enumerate(self.val_loader):
                src_input, tar_input, tar_output = data
                output, _ = model(src_input, tar_input)
                loss, accuracy, ppl = self.loss_accuracy(output, tar_output)
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_ppl += ppl
                count += 1

            test_input = src_input[0].unsqueeze(0)
            greedy_dec_input = greedy_decoder(model, test_input, seq_len=self.args.sequence_size)
            output, _ = model(test_input, greedy_dec_input)
            indices = output.view(-1, output.size(-1)).max(-1)[1].tolist()
            a = src_input[0].tolist()
            b = tar_output[0].tolist()
            output_sentence = self.tensor2sentence_di(indices)
            target_sentence = self.tensor2sentence_di(b)
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

    def model_save(self, model, epoch, step):
        model_name = '{0:06d}_transformer.pth'.format(step)
        model_path = os.path.join(self.args.model_path, model_name)
        torch.save({
            'epoch': epoch,
            'steps': step,
            'seq_len': self.args.sequence_size,
            'encoder_parameter': self.encoder_parameter(),
            'decoder_parameter': self.decoder_parameter(),
            'model_state_dict': model.state_dict()
        }, model_path)

    def tensor2sentence_di(self, indices: torch.Tensor) -> list:
        result = []
        translation_sentence = []
        for idx in indices:
            word = self.di_voc.IdToPiece(idx)
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


class Evaluation(object):
    def __init__(self, checkpoint, dictionary_path, x_test_path, y_test_path, file_name, batch_size=1,
                 beam_search=False, k=3):
        self.checkpoint = torch.load(checkpoint)
        self.max_seq = self.checkpoint['seq_len']
        self.ko_voc, self.en_voc = create_or_get_voca(save_path=dictionary_path)
        self.batch_size = batch_size
        self.x_test_path = x_test_path
        self.y_test_path = y_test_path
        self.file_name = 'test/' + file_name
        self.test_loader = self.get_test_loader()
        self.beam_search = beam_search
        self.k = k
        if beam_search:
            self.beam = Beam(beam_size=k, seq_len=self.max_seq)

    def model_load(self):
        encoder = Encoder(**self.checkpoint['encoder_parameter'])
        decoder = Decoder(**self.checkpoint['decoder_parameter'])
        model = Transformer(encoder, decoder)
        model = nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        return model

    def test(self, model):
        f = open(self.file_name, 'w', encoding='UTF8')
        count = 0
        bleu_scores = 0
        for i, data in enumerate(self.test_loader):

            src_input, tar_input, tar_output = data
            test_input = src_input[0].unsqueeze(0)
            if self.beam_search:
                self.beam = Beam(beam_size=self.k, seq_len=self.max_seq)
                beam_dec_input = self.beam.beam_search_decoder(model, test_input).unsqueeze(0)
                output, _ = model(src_input, beam_dec_input)
            else:
                greedy_dec_input = greedy_decoder(model, test_input, seq_len=self.max_seq)
                output, _ = model(src_input, greedy_dec_input)
            indices = output.view(-1, output.size(-1)).max(-1)[1].tolist()
            a = src_input[0].tolist()
            b = tar_output[0].tolist()
            output_sentence = self.tensor2sentence_di(indices)
            target_sentence = self.tensor2sentence_di(b)
            bleu_score = n_gram_precision(output_sentence[0], target_sentence[0])
            print("-------test-------")
            print("Korean: ", self.tensor2sentence_ko(a))  # input 출력
            print("Predicted : ", output_sentence)  # output 출력
            print("Target :", target_sentence)  # target 출력
            print('BLEU Score : ', bleu_score)
            f.write("-------test-------\n")
            f.write("Korean: " + self.tensor2sentence_ko(a)[0] + "\n")
            f.write("Predicted : " + output_sentence[0] + "\n")
            f.write("Target :" + target_sentence[0] + "\n")
            f.write('BLEU Score : ' + str(bleu_score) + "\n")
            bleu_scores += bleu_score
            count += 1
        avg_bleu = bleu_scores / count
        print("Average BLEU Score: ", str(avg_bleu))
        f.write("Average BLEU Score: " + str(avg_bleu))
        f.close()
        return avg_bleu

    def get_test_loader(self):
        test_dataset = LSTMSeq2SeqDataset(self.x_test_path, self.y_test_path, self.ko_voc, self.en_voc,
                                          self.max_seq)
        # dataset을 인자로 받아 data를 뽑아냄
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return test_loader

    def tensor2sentence_di(self, indices: torch.Tensor) -> list:
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


class Translation(object):  # Usage
    def __init__(self, checkpoint, dictionary_path, beam_search=False, k=3):
        self.checkpoint = torch.load(checkpoint)
        self.seq_len = self.checkpoint['seq_len']
        self.beam_search = beam_search
        if beam_search:
            self.beam = Beam(beam_size=k, seq_len=self.seq_len)
        self.k = k
        self.ko_voc, self.en_voc = create_or_get_voca(save_path=dictionary_path)

    def model_load(self):
        encoder = Encoder(**self.checkpoint['encoder_parameter'])
        decoder = Decoder(**self.checkpoint['decoder_parameter'])
        model = Transformer(encoder, decoder)
        model = nn.DataParallel(model)
        model.cuda()
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        return model

    def src_input(self, sentence):
        idx_list = self.ko_voc.EncodeAsIds(sentence)
        idx_list = self.padding(idx_list, self.ko_voc['<pad>'])
        return torch.tensor([idx_list]).to(device)

    def padding(self, idx_list, padding_id):
        length = len(idx_list)
        if length < self.seq_len:
            idx_list = idx_list + [padding_id for _ in range(self.seq_len - len(idx_list))]
        else:
            idx_list = idx_list[:self.seq_len]
        return idx_list

    def korean2dialect(self, model, sentence: str) -> (str, torch.Tensor):
        enc_input = self.src_input(sentence)
        if self.beam_search:
            beam_dec_input = self.beam.beam_search_decoder(model, enc_input).unsqueeze(0)
            output, _ = model(enc_input, beam_dec_input)
        else:
            greedy_dec_input = greedy_decoder(model, enc_input, seq_len=self.seq_len)
            output, _ = model(enc_input, greedy_dec_input)
        indices = output.view(-1, output.size(-1)).max(-1)[1].tolist()
        a = enc_input[0].tolist()
        output_sentence = self.tensor2sentence_en(indices)
        print("Korean: ", self.tensor2sentence_ko(a))  # input 출력
        print("Predicted : ", output_sentence)  # output 출력
        return output_sentence

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


