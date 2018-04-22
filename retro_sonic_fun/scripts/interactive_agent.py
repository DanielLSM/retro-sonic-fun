import retro

ACTIONS = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
,[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
,[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
,[1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
,[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
,[1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]
,[0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]
,[1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]]
#  {null, left, right, left+down, right+down, down, down+jump, jump}

def main():
    env = retro.make(game='SonicTheHedgehog-Genesis',
		     state='GreenHillZone.Act1')
    obs = env.reset()
    sample_action = env.action_space.sample()

    while True:
        ac = -1
        while (0 > ac) or (ac > 8):
            try:
                ac = int(input("Enter the action:\n"))
            except:
                continue
        action = ACTIONS[ac]
        # action[ac] = 1
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
