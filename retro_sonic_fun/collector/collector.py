import numpy as np 
from common.utils import normalize_obs
import multiprocessing
import retro
import datetime

def main(args):
    process = multiprocessing.current_process().name
    env = retro.make(game=args.name, state='Level1')
    obs = env.reset()
    for batch in range(args.total_batchs):
        states,actions,rewards,dones = [],[],[],[]
        for episode in range(args.episodes_per_batch):
            done = False
            while not done:
                action = env.action_space.sample()
                obs, rew, done, info = env.step(action)
                states.append(obs)
                actions.append(action)
                rewards.append(rew)
                dones.append(done)
                if done:
                    obs = env.reset()
        np.save('./data/{}/{}/{}_batch_{}/states'.format(args.game, args.now, process, batch), states)
        np.save('./data/{}/{}/{}_batch_{}/actions'.format(args.game, args.now, process, batch), actions)
        np.save('./data/{}/{}/{}_batch_{}/rewards'.format(args.game, args.now, process, batch), rewards)
        np.save('./data/{}/{}/{}_batch_{}/dones'.format(args.game, args.now, process, batch), dones)
        print("batch {} from {} done".format(batch, process))


if __name__ == '__main__':
    parser.add_argument('--game', type=str, default = 'Airstriker-Genesis', help='total number of batches')
    parser.add_argument('--total_batchs', type=int, default = 200, help='total number of batches')
    parser.add_argument('--episodes_per_batch', type=int, default = 200, help='total number of episodes per batch')
    parser.add_argument('--num_cpus', type=int, default = 1, help='total number of cpus')
    args = parser.parse_args()

    args.total_batchs %= args.num_cpus

    now = datetime.datetime.now()
    args.now = now.strftime("%Y-%m-%d %H:%M")
    p = multiprocessing.Pool(process=args.num_cpus)
    for _ in range(int(args.num_cpus)):
        p.apply_async(main, args=args)
    # main(args)