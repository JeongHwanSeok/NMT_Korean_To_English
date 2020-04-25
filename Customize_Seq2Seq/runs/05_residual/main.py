import argparse
from Customize_Seq2Seq.tools import Trainer, Translation


def get_args():
    parser = argparse.ArgumentParser()
    # 1. File path
    parser.add_argument('--data_path', default='../Data', type=str)
    parser.add_argument('--dictionary_path', default='../Dictionary', type=str)
    parser.add_argument('--src_train_filename', default='ko.train', type=str)
    parser.add_argument('--tar_train_filename', default='je.train', type=str)
    parser.add_argument('--src_val_filename', default='ko.dev', type=str)
    parser.add_argument('--tar_val_filename', default='je.dev', type=str)
    parser.add_argument('--model_path', default='Model', type=str)

    # 2. Model Hyper Parameter
    # 임베딩의 차원 rnn의 차원들을 전부 통일 시켜줘야함
    parser.add_argument('--sequence_size', default=50, type=int)
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--embedding_size', default=4000, type=int)

    # 3. Eecoder
    parser.add_argument('--encoder_rnn_dim', default=256, type=int)
    parser.add_argument('--encoder_n_layers', default=3, type=int)
    # dropout : drop하는 노드의 비율
    parser.add_argument('--encoder_embedding_dropout', default=0, type=float)
    parser.add_argument('--encoder_rnn_dropout', default=0, type=float)
    parser.add_argument('--encoder_dropout', default=0, type=float)
    parser.add_argument('--encoder_residual_used', default=True, type=bool)
    parser.add_argument('--encoder_bidirectional_used', default=True, type=float)
    parser.add_argument('--encoder_output_transformer', default=300)
    parser.add_argument('--encoder_output_transformer_bias', default=True, type=bool)
    parser.add_argument('--encoder_hidden_transformer', default=300)
    parser.add_argument('--encoder_hidden_transformer_bias', default=True, type=bool)

    # 4. Decoder
    parser.add_argument('--decoder_rnn_dim', default=256, type=int)
    parser.add_argument('--decoder_n_layers', default=3, type=int)
    # dropout : drop하는 노드의 비율
    parser.add_argument('--decoder_embedding_dropout', default=0, type=float)
    parser.add_argument('--decoder_dropout', default=0, type=float)
    parser.add_argument('--decoder_rnn_dropout', default=0, type=float)
    parser.add_argument('--decoder_residual_used', default=True, type=bool)

    # 5. learning hyper parameter
    parser.add_argument('--learning_method', default='Teacher_Forcing', type=str,
                        choices=['Teacher_Forcing', 'Scheduled_Sampling'])
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--plot_count', default=6, type=int)
    parser.add_argument('--train_step_print', default=10, type=int)
    parser.add_argument('--val_step_print', default=100, type=int)
    parser.add_argument('--step_save', default=1000, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    Trainer(args)


