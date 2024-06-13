"""

Implements a single forward pass.

"""

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece as spm
import transformer

# spm_processor = spm.SentencePieceProcessor

# pad_id = spm_processor.pad_id()
def get_attention_mask_and_positions(example: jax.Array,
                                     pad_id : int,
                                     )-> tuple[jax.Array, jax.Array]:
  """Builds the position and attention mask vectors from the given tokens."""
  pad_mask = example != pad_id
  current_token_position = transformer.build_positions_from_mask(pad_mask)
  attention_mask = transformer.make_causal_attn_mask(pad_mask)
  return current_token_position, attention_mask


def forward_pass(params,
                 *,
            model: transformer.Transformer,
            input_tokens: jax.Array,            # Shape [B, L]
            input_mask: jax.Array,              # Shape [B, L]
            positions: jax.Array,               # Shape [B, L]
            attention_mask: jax.Array,          # [B, L, L]
            ) -> jax.Array:
  """Foward pass.

  Args:
    params: model's input parameters.
    model: gemma transformer model to call.
    input_tokens: input tokens sequence, shape [B, L].
    input_mask: tokens to ignore when computing the loss, shape [B, L].
    positions: relative position of each token, shape [B, L].
    attention_mask: input attention mask, shape [B, L].

  Returns:
    Softmax cross-entropy loss for the next-token prediction task.
  """

  # Foward pass on the input data.
  # No attention cache is needed here.
  logits, _ = model.apply(
        params,
        input_tokens,
        positions,
        None,              # Attention cache is None.
        attention_mask,
    )

  return logits



vocab_path = "tokenizer.model" #set vocab path according to your file structure
model_path = "" # set model path

vocab = spm.SentencePieceProcessor()
vocab.Load(vocab_path)

pad_id = vocab.piece_to_id('<pad>')


text_input = "Human: Would you harm humans ? Assistant: "

# tokenize input
tokens = jnp.array([vocab.bos_id()] + vocab.EncodeAsIds(text_input))

positions, attention_mask = get_attention_mask_and_positions(tokens,pad_id)



linear_func, jvp_func = jax.linearize(lambda p: forward_pass(p, positions,tokens,attention_mask), param_diff)