import jax.numpy as jnp
from jax import linearize
from gemma import params as params_lib
import sentencepiece as spm
from gemma import transformer as transformer_lib


CKPT_PATH = " path to model weights"
TOKENIZER_PATH = "path to tokenizer"


vocab = spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)

GEMMA_VARIANT = '2b-it' # @param ['2b', '2b-it'] {type:"string"}

params = params_lib.load_and_format_params(CKPT_PATH)

transformer_config = transformer_lib.TransformerConfig.from_params(
    params=params,
    cache_size=1024
)

transformer = transformer_lib.Transformer(transformer_config)

logits, f_jvp = linearize(transformer, params['transformer']) # call linearize method

gemma_linear = logits + f_jvp(param_diff,x_zeros) # define gemma_linear



# write code to sample from linear gemma model


# compare outputs of linearized transformer model and finetuned model

    # estimate val loss

    # print out a set of outputs and compare both. 






