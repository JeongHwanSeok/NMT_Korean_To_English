import argparse
from Customize_Seq2SeqWithAttention.tools import Trainer, Translation


def get_args():
    parser = argparse.ArgumentParser()
    # 1. File path
    parser.add_argument('--data_path', default='../Data/jeju', type=str)
    parser.add_argument('--dictionary_path', default='../Dictionary/jeju', type=str)
    parser.add_argument('--src_train_filename', default='ko.train', type=str)
    parser.add_argument('--tar_train_filename', default='je.train', type=str)
    parser.add_argument('--src_val_filename', default='ko.dev', type=str)
    parser.add_argument('--tar_val_filename', default='je.dev', type=str)
    parser.add_argument('--model_path', default='Model', type=str)
    parser.add_argument('--img_path', default='img', type=str)

    # 2. Model Hyper Parameter
    # 임베딩의 차원 rnn의 차원들을 전부 통일 시켜줘야함
    parser.add_argument('--sequence_size', default=50, type=int)
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--embedding_size', default=4000, type=int)

    # 3. Eecoder
    parser.add_argument('--encoder_rnn_dim', default=256, type=int)
    parser.add_argument('--encoder_n_layers', default=3, type=int)
    # dropout : drop하는 노드의 비율
    parser.add_argument('--encoder_embedding_dropout', default=0.3, type=float)
    parser.add_argument('--encoder_rnn_dropout', default=0.3, type=float)
    parser.add_argument('--encoder_dropout', default=0.3, type=float)
    parser.add_argument('--encoder_residual_used', default=True, type=bool)
    parser.add_argument('--encoder_bidirectional_used', default=True, type=float)
    parser.add_argument('--encoder_output_transformer', default=256)
    parser.add_argument('--encoder_output_transformer_bias', default=True, type=bool)
    parser.add_argument('--encoder_hidden_transformer', default=256)
    parser.add_argument('--encoder_hidden_transformer_bias', default=True, type=bool)

    # 4. Decoder
    parser.add_argument('--decoder_rnn_dim', default=256, type=int)
    parser.add_argument('--decoder_n_layers', default=3, type=int)
    # dropout : drop하는 노드의 비율
    parser.add_argument('--decoder_embedding_dropout', default=0.3, type=float)
    parser.add_argument('--decoder_dropout', default=0.3, type=float)
    parser.add_argument('--decoder_rnn_dropout', default=0.3, type=float)
    parser.add_argument('--decoder_residual_used', default=True, type=bool)

    # 5. Attention
    parser.add_argument('--attention_score', default='general', type=str, choices=['dot', 'general'])
    parser.add_argument('--get_attention', default=True, type=bool)

    # 6. learning hyper parameter
    parser.add_argument('--learning_method', default='Scheduled_Sampling', type=str,
                        choices=['Teacher_Forcing', 'Scheduled_Sampling'])
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--plot_count', default=6, type=int)
    parser.add_argument('--train_step_print', default=10, type=int)
    parser.add_argument('--val_step_print', default=100, type=int)
    parser.add_argument('--step_save', default=1000, type=int)

    # 7. load model
    parser.add_argument('--model_load', default=False, type=bool)
    parser.add_argument('--checkpoint', default='Model/133000_model_1.pth', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    Trainer(args)

    # translation = Translation(
    #     checkpoint='Model/133000_model_1.pth',
    #     dictionary_path='../Dictionary',
    #     x_path='../Data/test.ko',
    #     y_path='../Data/test.en',
    #     beam_search=False,
    #     k=3,
    #     get_attention=True
    # )
    # translation.transform('세종대왕은 조선의 4대 왕이다.')
    # translation.batch_transform()

