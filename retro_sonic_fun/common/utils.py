import json


#normalize pixel values
def normalize_obs(obs):
    return obs.astype('float32') / 255.


def create_json_params(params: dict, model_name: str):
    file_name = '{}.json'.format(model_name)
    with open(file_name, 'w') as handle:
        json.dump(params, handle)
    print("{} generated".format(file_name))


def load_json_params(file_name: str):
    with open(file_name, 'r') as handle:
        params = json.load(handle)
    print("{} loaded".format(file_name))
    return params


class NN(type):
    pass
