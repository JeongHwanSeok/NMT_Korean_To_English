import argparse
from Transformer.tools import Trainer, Evaluation, Translation
import time


def get_args():
    parser = argparse.ArgumentParser()
    # 1. File path
    parser.add_argument('--data_path', default='../Data/gyeong', type=str)
    parser.add_argument('--dictionary_path', default='../Dictionary/gyeong', type=str)
    parser.add_argument('--src_train_filename', default='ko.train', type=str)
    parser.add_argument('--tar_train_filename', default='gy.train', type=str)
    parser.add_argument('--src_val_filename', default='ko.test', type=str)
    parser.add_argument('--tar_val_filename', default='gy.test', type=str)
    parser.add_argument('--model_path', default='Model', type=str)
    parser.add_argument('--img_path', default='img', type=str)

    # 2. Model Hyper Parameter
    parser.add_argument('--sequence_size', default=50, type=int)
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    # 3. Eecoder
    parser.add_argument('--encoder_vocab_size', default=4000, type=int)
    parser.add_argument('--encoder_hidden_dim', default=512, type=int)
    parser.add_argument('--encoder_layers', default=6, type=int)
    parser.add_argument('--encoder_heads', default=4, type=int)
    parser.add_argument('--encoder_head_dim', default=64, type=int)
    parser.add_argument('--encoder_pf_dim', default=512, type=int)
    parser.add_argument('--encoder_dropout', default=0.3, type=float)
    # 4. Decoder
    parser.add_argument('--decoder_vocab_size', default=4000, type=int)
    parser.add_argument('--decoder_hidden_dim', default=512, type=int)
    parser.add_argument('--decoder_layers', default=6, type=int)
    parser.add_argument('--decoder_heads', default=4, type=int)
    parser.add_argument('--decoder_head_dim', default=64, type=int)
    parser.add_argument('--decoder_pf_dim', default=512, type=int)
    parser.add_argument('--decoder_dropout', default=0.3, type=float)

    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--early_stopping', default=50, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--plot_count', default=6, type=int)
    parser.add_argument('--train_step_print', default=10, type=int)
    parser.add_argument('--val_step_print', default=100, type=int)
    parser.add_argument('--step_save', default=1000, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # args = get_args()
    # Trainer(args)

    # -------evaluate-------
    # start = time.time()
    # evaluate = Evaluation(checkpoint='Model/gyeong/best_transformer.pth', dictionary_path='../Dictionary/gyeong',
    #                 x_test_path='../Data/gyeong/ko.test', y_test_path='../Data/gyeong/gy.test',
    #                 file_name='gyeong_beam_k_temp.txt', beam_search=False, k=5)
    # model = evaluate.model_load()
    # test = evaluate.test(model)
    # end = time.time() - start
    # print("time: ", str(end))

    # -------predict-------
    # translation = Translation(checkpoint='Model/gyeong/best_transformer.pth', dictionary_path='../Dictionary/gyeong',
    #                           beam_search=True, k=5)
    # model = translation.model_load()
    # sentences = ["슈퍼마켓에 일주일에 한번 가요."
    #              ]
    # start = time.time()
    # for i in sentences:
    #     output = translation.korean2dialect(model, i)
    # end = time.time() - start
    # print("time: ", str(end))

    # -------batch predict-------
    translation = Translation(checkpoint='Model/gyeong/best_transformer.pth', dictionary_path='../Dictionary/gyeong',
                              x_test_path='translation/구어체2.test', file_name='translation/구어체2.csv',
                              beam_search=True, k=5)
    model = translation.model_load()
    start = time.time()
    output = translation.batch_korean2dialect(model)
    end = time.time() - start
    print("time: ", str(end))

