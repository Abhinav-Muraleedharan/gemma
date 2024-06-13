import os
import kagglehub
import jax.numpy as jnp
import sentencepiece as spm
import enum
import re
import string
import chex
import jax
import kagglehub
from jax import linearize
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib



os.environ["KAGGLE_USERNAME"] = "iamabhinavm"
os.environ["KAGGLE_KEY"] = "1c2584a3892091c9a536525da0d4b0ba"

GEMMA_VARIANT = '2b' # @param ['2b', '2b-it'] {type:"string"}
GEMMA_VARIANT_2 = '2b-it' # @param ['2b', '2b-it'] {type:"string"}



GEMMA_PATH = kagglehub.model_download(f'google/gemma/flax/{GEMMA_VARIANT}')
GEMMA_PATH_2 = kagglehub.model_download(f'google/gemma/flax/{GEMMA_VARIANT_2}')



CKPT_PATH = os.path.join(GEMMA_PATH, GEMMA_VARIANT)
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')

CKPT_PATH_2 = os.path.join(GEMMA_PATH_2, GEMMA_VARIANT_2)
TOKENIZER_PATH_2 = os.path.join(GEMMA_PATH_2, 'tokenizer.model')


vocab = spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)

GEMMA_VARIANT = '2b-it' # @param ['2b', '2b-it'] {type:"string"}

params = params_lib.load_and_format_params(CKPT_PATH) # load base  model params
params_2 = params_lib.load_and_format_params(CKPT_PATH_2) # load finetuned model parameters


config_2b = transformer_lib.TransformerConfig.from_params(
    params,
    cache_size=1024
)

model_2b = transformer_lib.Transformer(config=config_2b)




# helper function to compute difference between parameters
def diff_params(params_1,params_2):
    # Use tree_map to apply the function element-wise to the pytrees
    pytree_diff = jax.tree_util.tree_map(diff_fn, params_1, params_2)
    return pytree_diff
    
def diff_fn(x, y):
    return x - y

def get_attention_mask_and_positions(example: jax.Array,
                                     pad_id : int,
                                     )-> tuple[jax.Array, jax.Array]:
  """Builds the position and attention mask vectors from the given tokens."""
  pad_mask = example != pad_id
  current_token_position = transformer_lib.build_positions_from_mask(pad_mask)
  attention_mask = transformer_lib.make_causal_attn_mask(pad_mask)
  return current_token_position, attention_mask


def forward_and_loss_fn(params,
                        *,
                        model: transformer_lib.Transformer,
                        input_tokens: jax.Array,            # Shape [B, L]
                        input_mask: jax.Array,              # Shape [B, L]
                        positions: jax.Array,               # Shape [B, L]
                        attention_mask: jax.Array,          # [B, L, L]
                        ) -> jax.Array:
  """The forward pass and the loss function.

  Args:
    params: Model's input parameters.
    model: The Gemma transformer model to call.
    input_tokens: Input tokens sequence, shape [B, L].
    input_mask: Tokens to ignore when computing the loss, shape [B, L].
    positions: Relative position of each token, shape [B, L].
    attention_mask: Input attention mask, shape [B, L].

  Returns:
    The softmax cross-entropy loss for the next-token prediction task.
  """
  model= model_2b
  # The forward pass on the input data.
  # No attention cache is needed here.
  logits, _ = model.apply(
        params,
        input_tokens,
        positions,
        None,              # Attention cache is None.
        attention_mask,
    )

  # Exclude the last step as it does not appear in the targets.
  logits = logits[0, :-1]

  # Similarly, the first token cannot be predicted.
  target_tokens = input_tokens[0, 1:]
  target_mask = input_mask[0, 1:]

  # Convert the target labels to one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[...,None]

  # Define the normalization factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the negative log likelihood (NLL) loss.
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor

def forward_pass_logit_fn(params,
                        input_tokens: jax.Array,            # Shape [B, L]
                        positions: jax.Array,               # Shape [B, L]
                        attention_mask: jax.Array,          # [B, L, L]
                        ) -> jax.Array:
  """The forward pass and the loss function.

  Args:
    params: Model's input parameters.
    model: The Gemma transformer model to call.
    input_tokens: Input tokens sequence, shape [B, L].
    input_mask: Tokens to ignore when computing the loss, shape [B, L].
    positions: Relative position of each token, shape [B, L].
    attention_mask: Input attention mask, shape [B, L].

  Returns:
    The softmax cross-entropy loss for the next-token prediction task.
  """
  model= model_2b
  # The forward pass on the input data.
  # No attention cache is needed here.
  logits, _ = model.apply(
        {'params': params},
        input_tokens,
        positions,
        None,              # Attention cache is None.
        attention_mask,
    )

  # Exclude the last step as it does not appear in the targets.
#   logits = logits[0, :]
  return logits 





diff_params = diff_params(params['transformer'],params_2['transformer'])
del params_2 #unload params_2 after computing diff

pad_id = vocab.piece_to_id('<pad>')


text_input = "Human: Would you harm humans ? Assistant:  "

# tokenize input
input_tokens = jnp.array([vocab.bos_id()] + vocab.EncodeAsIds(text_input))
positions, attention_mask = get_attention_mask_and_positions(input_tokens,pad_id)
reshaped_tokens = input_tokens.reshape((1, 21))
positions, attention_mask = get_attention_mask_and_positions(input_tokens, pad_id)

# check if forward pass working:
logits = forward_pass_logit_fn(params=params['transformer'],input_tokens = reshaped_tokens,positions= positions,attention_mask= attention_mask)

#linearization code:
y,f_jvp = jax.linearize(lambda p: forward_pass_logit_fn(p,input_tokens = reshaped_tokens,positions= positions,attention_mask= attention_mask), params['transformer'])

logit_linear_approx = y + f_jvp(diff_params)


