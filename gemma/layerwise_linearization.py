
import os
import gemma 
import jax 
import jax.numpy as jnp
import kagglehub
from gemma import params as params_lib
from flax import linen as nn
from jax import jvp

def feedforward(x,params):
    """
    Feed Forward Layer definition 
    for gemma transformer model.
    Adopted from layers.py
    
    """
    w_0 = params[0]
    w_1 = params[1]
    w_2 = params[2]
    ff_gate = jnp.dot(x, w_0)
    gate_value = nn.gelu(ff_gate)

    ff1 = jnp.dot(x, w_1)
    activations = gate_value * ff1

    w_linear = w_2
    outputs = jnp.dot(activations, w_linear)

    return outputs

def diff(params_1,params_2):
    params_3 = params_1 - params_2
    return params_3



GEMMA_VARIANT = '2b-it' # Instruction finetuned Gemma
GEMMA_PATH = kagglehub.model_download(f'google/gemma/flax/{GEMMA_VARIANT}')


GEMMA_VARIANT_2 = '2b' # Base Model
GEMMA_PATH_2 = kagglehub.model_download(f'google/gemma/flax/{GEMMA_VARIANT_2}')



CKPT_PATH = os.path.join(GEMMA_PATH, GEMMA_VARIANT)
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')
print('CKPT_PATH:', CKPT_PATH)
print('TOKENIZER_PATH:', TOKENIZER_PATH)

CKPT_PATH_2 = os.path.join(GEMMA_PATH_2, GEMMA_VARIANT_2)
TOKENIZER_PATH_2 = os.path.join(GEMMA_PATH_2, 'tokenizer.model')
print('CKPT_PATH:', CKPT_PATH_2)
print('TOKENIZER_PATH:', TOKENIZER_PATH_2)



params_1 = params_lib.load_and_format_params(CKPT_PATH)
params_2 = params_lib.load_and_format_params(CKPT_PATH_2)

# extract mlp layers of 2 versions of gemma transformer
param_feedforward_1 = params_1['transformer']['layer_7']['mlp'] # extract layer parameters of finetuned model
param_feedforward_2 = params_2['transformer']['layer_7']['mlp'] # extract layer parameters of base model



params_1 = [param_feedforward_1['gating_einsum'][0],
            param_feedforward_1['gating_einsum'][1],
            param_feedforward_1['linear']]

params_2 = [param_feedforward_2['gating_einsum'][0],
            param_feedforward_2['gating_einsum'][1],
            param_feedforward_2['linear']]


# check if inference code is working:
x = jnp.ones(2048)
feedforward(x,params_1)

# Now we can use jax methods to evaluate jacobian vector products. 
y, u = jvp(feedforward, (x,params_1), (x,diff(params_1,params_2)))
