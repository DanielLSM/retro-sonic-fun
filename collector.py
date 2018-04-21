import numpy as np 
from common.utils import normalize_obs
import multiprocessing
import retro

def main(args):
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    obs = env.reset()
    for batch in range(total_batchs):
        #TODO it would prolly be better to initialize a large array than keep extending a list
        states,actions,rewards,dones = [],[],[],[]
        for episode in range(args.episodes_per_batch):
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                obs = env.reset()


if __name__ == '__main__':
    parser.add_argument('--total_batchs', type=int, default = 200, help='total number of batches')
    parser.add_argument('--episodes_per_batch', type=int, default = 200, help='total number of episodes per batch')
    parser.add_argument('--num_cpus', type=int, default = 1, help='total number of cpus')episodes to generate
    parser.add_argument('--render', action='store_true', help='render the env as data is generated')
    args = parser.parse_args()

    args.total_batchs %= args.num_cpus

    main(args)