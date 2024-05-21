import jax.numpy as jnp
from jax import linearize

logits, f_jvp = linearize(Gemma, params, x) # call linearize method
gemma_linear = logits + f_jvp(param_diff,x_zeros) # define gemma_linear

# write code to sample from linear gemma model


# compare outputs of linearized transformer model and finetuned model

    # estimate val loss

    # print out a set of outputs and compare both. 



