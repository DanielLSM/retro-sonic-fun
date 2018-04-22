import retro
from retro_sonic_fun.common.utils import SonicDiscretizer
import keyboard

actions = [
    ['LEFT'],  #0
    ['RIGHT'],  #1
    ['LEFT', 'DOWN'],  #2
    ['RIGHT', 'DOWN'],  #3
    ['DOWN'],  #4
    ['DOWN', 'B'],  #5
    ['B'],  #6
    ['RIGHT', 'B'],  #7
    ['LEFT', 'B']  #8
]


def main():
    env = retro.make(
        game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    action_controller = SonicDiscretizer(env)
    obs = env.reset()
    sample_action = env.action_space.sample()

    while True:
        ac = -1
        while (0 > ac) or (ac > 8):
            try:
                ac = int(input("Enter the action:\n"))
            except:
                continue

        action = action_controller.action(ac)
        for _ in range(10):
            obs, rew, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()


if __name__ == '__main__':
    main()
