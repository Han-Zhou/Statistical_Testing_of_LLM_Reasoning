# Patch: Flash Attention 2 for Qwen3.5 in transformers 5.3.0

## TL;DR

Two-line guard added to `_is_packed_sequence()` in the installed transformers
package so that Qwen3.5 stops crashing with `CUDA error: an illegal memory
access was encountered` when loaded with `attn_implementation="flash_attention_2"`.

This is a workaround for upstream issue
[huggingface/transformers#44910](https://github.com/huggingface/transformers/issues/44910).
The issue is marked Closed, but the fix is **not present in any released
transformers version, nor in the current `main` branch** as of 2026-05-22. We
are running `transformers==5.3.0`, which has the bug.

## File touched

```
/pm/miniconda3/envs/users/han/cot/lib/python3.10/site-packages/transformers/modeling_flash_attention_utils.py
```

This is a site-packages file inside the `cot` conda env. The change is **not
under version control** — if the env is rebuilt, the patch must be reapplied
(or replaced by upgrading transformers once an official fix lands).

## What the bug is

When generating with Qwen3.5 + FA2 + bf16, the model crashes inside
`_flash_attention_forward` at the line:

```python
elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
```

with `RuntimeError: CUDA error: an illegal memory access was encountered`.

### Root cause

`_flash_attention_forward` chooses between two code paths:

1. **Standard FA2** — one kernel call over the whole `[batch, seq_len, ...]`
   tensor. This is what almost every model uses.
2. **Variable-length FA2** (`flash_attn_varlen_func`) — used when several
   short sequences are *packed* into a single row, with `position_ids` that
   restart at each sub-sequence (e.g. `[0,1,2,3,0,1,2,0,1,2,3,4]`).
   It needs `cu_seqlens` to find the boundaries.

The dispatch is done by `_is_packed_sequence()`, which reads
`position_ids.shape[1]` to get the sequence length and checks whether the
positions form one monotonically increasing run.

Qwen3.5 uses **multi-axis rotary embeddings** and passes 3D `position_ids` of
shape `[n_axes, batch, seq_len]` (see e.g. `modeling_qwen3_5.py:1346`,
`:1348`, `:1702`, `:1705` — they unconditionally expand position ids to 3D).
For such a tensor `shape[1]` is the **batch** dimension (= 1), not `seq_len`.
The packed-sequence check then misfires and returns `True`, sending the model
down the varlen path with a bogus `cu_seqlens` like `[0, 256, 512, 768]`
(claiming three packed sequences totaling 768 tokens when q/k/v only contain
256). Flash Attention's CUDA kernel reads past the end of the buffers, and
the GPU reports an illegal memory access.

The bug is **not multi-GPU specific** — it would crash the same way on a
single GPU. The 2-GPU `device_map="auto"` setup we were running was incidental.

## What the fix does

Add a dimensionality guard at the top of `_is_packed_sequence()`. Packed
sequences are by definition 2D (`[batch=1, total_tokens]`); any tensor with
`dim() > 2` cannot be a packed sequence and should fall through to the
standard FA2 path.

### Before (transformers 5.3.0, lines 444-457)

```python
def _is_packed_sequence(position_ids, batch_size):
    """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False

    increasing_position_sequences = (
        torch.arange(position_ids.shape[1], device=position_ids.device) + position_ids.min()
    )
    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()
```

### After

```python
def _is_packed_sequence(position_ids, batch_size):
    """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False

    # Qwen3.5 (and other multi-axis-rope models) pass 3D position_ids of shape
    # [n_axes, batch, seq_len]. Packed sequences are always 2D; reject anything
    # higher-dim to avoid building bogus cu_seqlens from the wrong axis.
    # See https://github.com/huggingface/transformers/issues/44910
    if position_ids.dim() > 2:
        return False

    increasing_position_sequences = (
        torch.arange(position_ids.shape[1], device=position_ids.device) + position_ids.min()
    )
    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()
```

The added block is 6 lines (4 of which are an explanatory comment).

## Why this is safe

- Behavior for any model with 2D or 1D `position_ids` is identical — the new
  guard never triggers for them.
- The only path the guard alters is the 3D case, which the original function
  was never designed to handle correctly. Returning `False` there sends the
  model down the standard Flash Attention path, which is what multi-axis-rope
  models like Qwen3.5 actually want.
- This is the same fix proposed in the upstream issue's referenced branch
  (`ouroborosscr/transformers/tree/fix/qwen35-flash-attn-3d-position-ids`).

## How to undo / reapply

The patch lives in a single function; to revert, delete the 6 added lines
between `if position_ids is None: return False` and the
`increasing_position_sequences = ...` block.

If the conda env is recreated, reapply by editing the same file or by
installing the reporter's branch:

```
pip install "git+https://github.com/ouroborosscr/transformers.git@fix/qwen35-flash-attn-3d-position-ids"
```

Watch for the fix landing in an official transformers release; once it does,
upgrade and drop this note.

## Verification

Re-run `srun_qwen_flash.sh` (or `python3 temp_qwen_flash.py` directly on a
node with a GPU). The previous crash at `modeling_flash_attention_utils.py:677`
should be gone and generation should produce text.
