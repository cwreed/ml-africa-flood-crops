import sys
from argparse import ArgumentParser

sys.path.append('..')

from src.models import STR2MODEL, train_model

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_type', type=str, help='One of [cropland, flood]')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true')
    parser.add_argument('--override_early_stop', dest='early_stop', action='store_false')
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_known_args()[0]

    model_args = STR2MODEL[args.model_type].add_model_specific_args(parser).parse_args()
    model = STR2MODEL[args.model_type](model_args)

    train_model(model, model_args)