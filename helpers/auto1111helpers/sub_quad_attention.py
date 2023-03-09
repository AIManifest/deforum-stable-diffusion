from functools import partial
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, NamedTuple, List

#sub_quadratic_attention
class ScannedChunk(NamedTuple):
    chunk_idx: int
    attn_chunk: AttnChunk

def efficient_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_chunk_size=1024,
    kv_chunk_size: Optional[int] = None,
    kv_chunk_size_min: Optional[int] = None,
    use_checkpoint=True,
):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Args:
        query: queries for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        key: keys for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        value: values to be used in attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        query_chunk_size: int: query chunks size
        kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)
        kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
      Returns:
        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
      """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, k_tokens, _ = key.shape
    scale = q_channels_per_head ** -0.5

    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    def get_query_chunk(chunk_idx: int) -> Tensor:
        return narrow_trunc(
            query,
            1,
            chunk_idx,
            min(query_chunk_size, q_tokens)
        )
    
    summarize_chunk: SummarizeChunk = partial(_summarize_chunk, scale=scale)
    summarize_chunk: SummarizeChunk = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    compute_query_chunk_attn: ComputeQueryChunkAttn = partial(
        _get_attention_scores_no_kv_chunking,
        scale=scale
    ) if k_tokens <= kv_chunk_size else (
        # fast-path for when there's just 1 key-value chunk per query chunk (this is just sliced attention btw)
        partial(
            _query_chunk_attention,
            kv_chunk_size=kv_chunk_size,
            summarize_chunk=summarize_chunk,
        )
    )

    if q_tokens <= query_chunk_size:
        # fast-path for when there's just 1 query chunk
        return compute_query_chunk_attn(
            query=query,
            key=key,
            value=value,
        )
    
    # TODO: maybe we should use torch.empty_like(query) to allocate storage in-advance,
    # and pass slices to be mutated, instead of torch.cat()ing the returned slices
    res = torch.cat([
        compute_query_chunk_attn(
            query=get_query_chunk(i * query_chunk_size),
            key=key,
            value=value,
        ) for i in range(math.ceil(q_tokens / query_chunk_size))
    ], dim=1)
    return res
