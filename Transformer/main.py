import argparse
from Transformer.tools import Trainer, Translation


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
    parser.add_argument('--img_path', default='img', type=str)

    # 2. Model Hyper Parameter
    parser.add_argument('--sequence_size', default=50, type=int)
    parser.add_argument('--embedding_dim', default=512, type=int)
    parser.add_argument('--label_smoothing', default=0.1, type=float)

    # 3. Eecoder
    parser.add_argument('--encoder_vocab_size', default=4000, type=int)
    parser.add_argument('--encoder_hidden_dim', default=512, type=int)
    parser.add_argument('--encoder_layers', default=5, type=int)
    parser.add_argument('--encoder_heads', default=4, type=int)
    parser.add_argument('--encoder_head_dim', default=64, type=int)
    parser.add_argument('--encoder_pf_dim', default=512, type=int)
    parser.add_argument('--encoder_dropout', default=0.3, type=float)
    # 4. Decoder
    parser.add_argument('--decoder_vocab_size', default=4000, type=int)
    parser.add_argument('--decoder_hidden_dim', default=512, type=int)
    parser.add_argument('--decoder_layers', default=5, type=int)
    parser.add_argument('--decoder_heads', default=4, type=int)
    parser.add_argument('--decoder_head_dim', default=64, type=int)
    parser.add_argument('--decoder_pf_dim', default=512, type=int)
    parser.add_argument('--decoder_dropout', default=0.3, type=float)

    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--early_stopping', default=7, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--plot_count', default=6, type=int)
    parser.add_argument('--train_step_print', default=10, type=int)
    parser.add_argument('--val_step_print', default=100, type=int)
    parser.add_argument('--step_save', default=1000, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    Trainer(args)

    # translation = Translation(checkpoint='Model/006000_transformer.pth', dictionary_path='../Dictionary',
    #                           beam_search=False)
    # translation.korean2dialect("오누이가 학교를 다녀왔는데 그걸 친 걸 말 안한 거야 .")

