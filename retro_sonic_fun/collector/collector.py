import numpy as np
# from common.utils import normalize_obs
import multiprocessing
import retro
import os
import datetime
import argparse
from retro_sonic_fun.controller.models import RandomModel


def get_model(alg, env):
    if alg == 'random':
        return RandomModel(env)
    elif alg =='jerk';
        return JerkModel(env)
    else:
        raise NotImplementedError


def generate_data(args):
    process = multiprocessing.current_process().name
    env = retro.make(game=args.game, state=args.state)
    obs = env.reset()
    model = get_model(args.algorithm, env)
    names = ['states', 'actions', 'rewards', 'dones']
    for batch in range(args.total_batchs):
        arr = [[] for _ in range(4)]
        for episode in range(args.episodes_per_batch):
            done = False
            while not done:
                action = model.get_action(obs)
                obs, rew, done, _ = env.step(action)
                if args.render:
                    env.render()
                arr[0].append(obs)
                arr[1].append(action)
                arr[2].append(rew)
                arr[3].append(done)
                if done:
                    obs = env.reset()
            file_to_save = './data/{}/{}/{}/{}_batch_{}'.format(
                args.game, args.state, args.now,  process, batch)
            os.makedirs(os.path.dirname(file_to_save), exist_ok=True)
            np.savez_compressed(
                file_to_save,
                states=arr[0],
                actions=arr[1],
                rewards=arr[2],
                dones=arr[3])
        print("batch {} from {} done".format(batch, process))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--game',
        type=str,
        default='SonicTheHedgehog-Genesis',
        help='game to collect data from')
    parser.add_argument(
        '--state',
        type=str,
        default='GreenHillZone.Act1',
        help='specific level of the game to collect data from')
    parser.add_argument(
        '--total_batchs',
        '-tb',
        type=int,
        default=1,
        help='total number of batches')
    parser.add_argument(
        '--episodes_per_batch',
        '-eb',
        type=int,
        default=1,
        help='total number of episodes per batch')
    parser.add_argument(
        '--num_cpus', type=int, default=1, help='total number of cpus')
    parser.add_argument(
        '--algorithm',
        '-alg',
        type=str,
        default='random',
        help="algorithm used by agent")
    parser.add_argument(
        '--render', type=bool, default=False, help="to render or not")
    args = parser.parse_args()
    args.total_batchs //= args.num_cpus

    now = datetime.datetime.now()
    args.now = now.strftime("%Y-%m-%d_%H:%M")
    p = multiprocessing.Pool(processes=args.num_cpus)
    processes = [
        p.apply_async(generate_data, args=(args, ))
        for _ in range(args.num_cpus)
    ]
    for process in processes:
        process.get()
