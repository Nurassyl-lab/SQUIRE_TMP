import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/kinshiphinton_final/rules.dict')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    d = json.load(open(args.path))
    print('\trelations:', len(d), '| total rules:', sum(len(v) for v in d.values()))