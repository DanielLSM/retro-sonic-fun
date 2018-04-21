import numpy as np
from common.utils import normalize_obs
import multiprocessing
import retro
import os
import datetime
import argparse

def generate_data(args):
    process = multiprocessing.current_process().name
    env = retro.make(game=args.game, state='Level1')
    obs = env.reset()
    names = ['states', 'actions', 'rewards', 'dones']
    for batch in range(args.total_batchs):
        arr = [[] for _ in range(4)]
        for episode in range(args.episodes_per_batch):
            done = False
            while not done:
                action = env.action_space.sample()
                obs, rew, done, _ = env.step(action)
                arr[0].append(obs)
                arr[1].append(action)
                arr[2].append(rew)
                arr[3].append(done)
                if done:
                    obs = env.reset()
        for name, arr_to_save in zip(names, arr):
            file_to_save = './data/{}/{}/{}_batch_{}'.format(args.game,
                                                                args.now,
                                                                process,
                                                                batch)
            os.makedirs(os.path.dirname(file_to_save), exist_ok=True)
            # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.savez_compressed.html
            np.savez_compressed(file_to_save, states=arr[0], actions=arr[1],
                                rewards=arr[2], dones=arr[3])
        print("batch {} from {} done".format(batch, process))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', type=str, default = 'Airstriker-Genesis', help='total number of batches')
    parser.add_argument('--total_batchs', type=int, default = 200, help='total number of batches')
    parser.add_argument('--episodes_per_batch', type=int, default = 20, help='total number of episodes per batch')
    parser.add_argument('--num_cpus', type=int, default = 1, help='total number of cpus')
    args = parser.parse_args()

    args.total_batchs //= args.num_cpus

    now = datetime.datetime.now()
    args.now = now.strftime("%Y-%m-%d_%H:%M")
    p = multiprocessing.Pool(processes=args.num_cpus)
    processes = [p.apply_async(generate_data, args=(args,))
               for _ in range(args.num_cpus)]
    for process in processes:
        process.get()
