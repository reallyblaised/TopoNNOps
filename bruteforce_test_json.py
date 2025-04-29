import json
import torch.nn as nn
import numpy as np
import torch
from monotonicnetworks import GroupSort, direct_norm, MonotonicWrapper

def load_lipnn_model(filename, pair_indices=False):
    with open(filename, 'r') as f:
        input_file = json.load(f)
    print(input_file.keys())

    try:
        sigma = input_file['sigmanet.sigma'][0]
        print(sigma)
    except:
        print("No sigma found")
        sigma = 2.0
        print(f"Defaulting to {sigma}")

    # # SANITY CHECK
    # sigma = 2.0

    print("Sanity check sigma", sigma)
    n_layers = len(list(filter(lambda x: 'weight.original' in x, input_file.keys())))
    try:
        constraints = input_file['constraints']
    except:
        print("No constraints found")
        constraints = [1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0] #  [1,0,1,0,0,0,0,0,0] #
        print(f"Defaulting to {constraints}")
    print("Sanity check constraints", constraints)
    DEPTH = n_layers - 1

    # # activation = nn.Softmax()

    if 'sigmanet.nn.00.parametrizations.weight.original' in input_file.keys():
        suffix = '.original'
    else:
        suffix = ''

    first_weight = np.array(
        input_file[f'sigmanet.nn.00.parametrizations.weight{suffix}'], dtype='float32')

    first_bias = np.array(input_file['sigmanet.nn.00.bias'], dtype='float32')
    model = nn.Sequential()

    mta = nn.Linear(first_weight.shape[1], first_weight.shape[0])
    mta.weight.data = torch.from_numpy(first_weight)
    mta.bias.data = torch.from_numpy(first_bias)


    # mta = direct_norm(mta, kind='one-inf', max_norm=sigma)
    print(f"Booked first layer with kind: one-inf")
    model.add_module('0', mta)
    model.add_module('0-activ', GroupSort(first_weight.shape[0]//2))
    for layer in range(1, DEPTH):

        layer_idx = f"0{layer * 2}" if layer < 5 else f"{layer * 2}"

        layer_bias = np.array(
            input_file[f'sigmanet.nn.{layer_idx}.bias'], dtype='float32')
        layer_weight = np.array(
            input_file[f'sigmanet.nn.{layer_idx}.parametrizations.weight{suffix}'], dtype='float32')
        mta = nn.Linear(layer_weight.shape[1], layer_weight.shape[0])
        mta.weight.data = torch.from_numpy(layer_weight)
        mta.bias.data = torch.from_numpy(layer_bias)
        # mta = direct_norm(mta, kind='inf', max_norm=sigma)
        print(f"Booked layer {layer} with kind: inf")
        model.add_module(f'{layer}', mta)
        model.add_module(f'{layer}-activ', GroupSort(layer_weight.shape[0]//2))

    DEPTH *= 2
    last_weight = np.array(
        input_file[f'sigmanet.nn.{DEPTH}.parametrizations.weight{suffix}'], dtype='float32')
    assert (last_weight.shape[0] == 1)
    mta = nn.Linear(last_weight.shape[1], last_weight.shape[0])
    mta.weight.data = torch.from_numpy(last_weight)
    mta.bias.data = torch.from_numpy(
        np.array(input_file[f'sigmanet.nn.{DEPTH}.bias'], dtype='float32'))
    #mta = direct_norm(mta, kind='one', max_norm=sigma)
    print(f"Booked last layer with kind: one")
    model.add_module(f'{DEPTH}', mta)

    model = MonotonicWrapper(model, sigma, constraints)

    #model_out = nn.Sequential(model, nn.Sigmoid())
    # # constraints = [0,0,0,0]
    # model_out = modelNN.sigmoid_nn(model, sigma, constraints)
    # rescale_min = np.float64(input_file['rescale_min'])
# rescale_max = np.float64(input_file['rescale_max'])
    # return model_out, [rescale_min, rescale_max]

    return model

if __name__ == "__main__":
    model = load_lipnn_model("prod_model_ThreeBody.json")
    input_vector = torch.tensor(
    # [
    #     0.0446527, 
    #     0.24048, 
    #     0.271239, 
    #     0.617939, 
    #     0.243267, 
    #     0.0357454, 
    #     0.70625, 
    #     0.480977, 
    #     0.853921
    # ], # twobody
    [
        0.29779, 0.255747, 0.697874, 0.230225, 0.525004, 
        0.616805, 0.164721, 0.477467, 0.631799, 0.141092, 
        0.369142, 0.545994, 0.0752091, 1.0, 0.891192, 
        0.392082, 0.549011
    ], # threebody
    dtype=torch.float32)

    print(model(input_vector.unsqueeze(0)))
    output = torch.sigmoid(model(input_vector.unsqueeze(0)))
    print(output)

    # # test read in
    from model_persistence import load_into_lipnn
    model = load_into_lipnn("/work/submit/blaised/TopoNNOps/mlruns/6/5f66c1bf2f264ff8914a85ae0ffe0954/artifacts/model_state_dict.pt")
    #model = load_into_lipnn("/work/submit/blaised/TopoNNOps/mlruns/5/13dca32e581546699047b0d90c97ac5f/artifacts/model_state_dict.pt") # threebody
    output = torch.sigmoid(model(torch.tensor(input_vector).unsqueeze(0)))
    print(f"Loaded model from pt output: {output}")