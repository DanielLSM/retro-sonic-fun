#normalize pixel values
def adjust_obs(obs):
    return obs.astype('float32') / 255.