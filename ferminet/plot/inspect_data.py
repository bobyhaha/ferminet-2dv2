import numpy as np

ckpt = np.load("ferminet_2025_06_23_15:59:59/qmcjax_ckpt_000785.npz", allow_pickle=True)
print(type(ckpt["params"])) #numpu.ndarray
params = ckpt["params"].item() #convert to dict
print(params.keys())
#dict_keys(['envelope', 'layers', 'orbital'])
print(params['envelope'][0].keys()) #contain a list of dict of pi and sigma
print(params['layers'].keys()) #input and stream
print(params['layers']['streams'][0].keys()) #double and single
print(len(params['layers']['streams'])) #layer num is 4
print((params['layers']['streams'][0]['single'].keys())) #b and w
print(type(params['layers']['streams'][0]['double']['w'])) #b and w
#<class 'jaxlib._jax.ArrayImpl'> means it's dealing with jax array
data = ckpt["data"].item()