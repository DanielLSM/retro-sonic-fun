#normalize pixel values
def normalize_obs(obs):
    return obs.astype('float32') / 255.