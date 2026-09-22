import logging
import os

import coloredlogs

from Coach import Coach
from intransitive.IntransitiveGame import Intransitive as Game
from intransitive.pytorch.NNet import NNetWrapper as nn
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--numIters', type=int, default=15)
parser.add_argument('--numEps', type=int, default=40)
parser.add_argument('--numMCTSSims', type=int, default=40)
parser.add_argument('--arenaCompare', type=int, default=20)
parser.add_argument('--cpuct', type=float, default=1.0)
parser.add_argument('--load_model', type=str, default='')
cli_args = parser.parse_args()

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

checkpoint_path = cli_args.load_model
checkpoint_folder, checkpoint_filename = os.path.split(checkpoint_path)
checkpoint_folder = checkpoint_folder + '/'

args = dotdict({
    'numIters': cli_args.numIters,
    'numEps': cli_args.numEps,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': cli_args.numMCTSSims,
    'arenaCompare': cli_args.arenaCompare,
    'cpuct': cli_args.cpuct,
    'checkpoint': './temp/',
    'load_model': cli_args.load_model,
    'load_folder_file': (checkpoint_folder, checkpoint_filename),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(9)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()