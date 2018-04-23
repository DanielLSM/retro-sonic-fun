import retro
from retro_sonic_fun.common.utils import SonicDiscretizer
import keyboard
import time
import curses
import argparse

actions = [
    ['LEFT'],  #0
    ['RIGHT'],  #1
    ['LEFT', 'DOWN'],  #2
    ['RIGHT', 'DOWN'],  #3
    ['DOWN'],  #4
    ['DOWN', 'B'],  #5
    ['B'],  #6
    ['RIGHT', 'B'],  #7
    ['LEFT', 'B'],  #8
    ['NULL']  #9
]

actions_mapping = {
    'KEY_LEFT': 0,
    'KEY_RIGHT': 1,
    'KEY_DOWN': 4,
    'KEY_UP': 6,
    'e': 7,
    'q': 8,
    'a': 2,
    'd': 3
}

# SonicTheHedgehog-Genesis,SpringYardZone.Act3
# SonicTheHedgehog-Genesis,SpringYardZone.Act2
# SonicTheHedgehog-Genesis,GreenHillZone.Act3
# SonicTheHedgehog-Genesis,GreenHillZone.Act1
# SonicTheHedgehog-Genesis,StarLightZone.Act2
# SonicTheHedgehog-Genesis,StarLightZone.Act1
# SonicTheHedgehog-Genesis,MarbleZone.Act2
# SonicTheHedgehog-Genesis,MarbleZone.Act1
# SonicTheHedgehog-Genesis,MarbleZone.Act3
# SonicTheHedgehog-Genesis,ScrapBrainZone.Act2
# SonicTheHedgehog-Genesis,LabyrinthZone.Act2
# SonicTheHedgehog-Genesis,LabyrinthZone.Act1
# SonicTheHedgehog-Genesis,LabyrinthZone.Act3


def main(win, game, state):
    env = retro.make(game=game, state=state)
    action_controller = SonicDiscretizer(env)
    obs = env.reset()
    sample_action = env.action_space.sample()

    win.nodelay(True)
    key = 9
    win.clear()
    win.addstr("Detected key:")

    while True:
        time.sleep(1. / 45)

        try:
            key = win.getkey()
            win.clear()
            win.addstr("Detected key:")
            win.addstr(str(key))
            if key != 9:
                key = actions_mapping[key]
        except Exception as e:
            pass

        if key in actions_mapping.values():
            action = action_controller.action(int(key))
            obs, rew, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--game',
        '-g',
        type=str,
        default='SonicTheHedgehog-Genesis',
        help='game name')
    parser.add_argument(
        '--state',
        '-s',
        type=str,
        default='SpringYardZone.Act2',
        help='name of the scene')
    args = parser.parse_args()
    curses.wrapper(main, args.game, args.state)
