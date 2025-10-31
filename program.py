from __future__ import annotations

import dataclasses
from typing import Dict, List

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, BlockSamplingProgram
from thrml.conditional_samplers import SoftmaxConditional, _State
from thrml.interaction import InteractionGroup
from thrml.pgm import CategoricalNode, DEFAULT_NODE_SHAPE_DTYPES

from .model import TextDiffusionModel


@dataclasses.dataclass(eq=False)
class InputTokenNode(CategoricalNode):
    position: int


@dataclasses.dataclass(eq=False)
class DiffusionTokenNode(CategoricalNode):
    step: int
    position: int


class TextDiffusionSampler(SoftmaxConditional):
    model: TextDiffusionModel
    step_index: int

    def compute_parameters(
        self,
        key: Key,
        interactions: List[PyTree],
        active_flags: List[Array],
        states: List[List[_State]],
        sampler_state,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ):
        del key, interactions, output_sd
        if len(states) != 1:
            raise RuntimeError("Expected a single interaction per block")
        tail_states = states[0]
        if len(tail_states) != 2:
            raise RuntimeError("Diffusion sampler expects previous and input states")
        prev_state = tail_states[0][:, 0].astype(jnp.int32)
        input_state = tail_states[1][:, 0].astype(jnp.int32)
        logits = self.model.compute_logits(prev_state, input_state, self.step_index)
        logits = logits.astype(jnp.float32)

        pad_mask = input_state == self.model.input_pad_id
        logits = jnp.where(pad_mask[:, None], -1e9, logits)
        logits = logits.at[:, self.model.target_pad_id].set(
            jnp.where(pad_mask, 0.0, logits[:, self.model.target_pad_id])
        )

        if active_flags:
            active = active_flags[0][:, 0].astype(bool)
            logits = jnp.where(active[:, None], logits, -1e9)

        return logits, sampler_state


@dataclasses.dataclass
class TextDiffusionProgramSpec:
    program: BlockSamplingProgram
    gibbs_spec: BlockGibbsSpec
    input_block: Block
    latent_blocks: List[Block]
    terminal_block: Block
    latent_block_indices: Dict[int, int]
    input_block_index: int
    terminal_block_index: int


def build_text_diffusion_program(model: TextDiffusionModel, seq_len: int) -> TextDiffusionProgramSpec:
    n_steps = int(model.step_embed.shape[0])
    input_nodes = [InputTokenNode(i) for i in range(seq_len)]
    diffusion_nodes = [
        [DiffusionTokenNode(step=t, position=i) for i in range(seq_len)]
        for t in range(n_steps + 1)
    ]

    input_block = Block(input_nodes)
    latent_blocks = [Block(layer) for layer in diffusion_nodes[:-1]]
    terminal_block = Block(diffusion_nodes[-1])

    free_super_blocks = list(reversed(latent_blocks))
    node_shape_dtypes = dict(DEFAULT_NODE_SHAPE_DTYPES)
    token_sd = jax.ShapeDtypeStruct(tuple(), dtype=jnp.int32)
    node_shape_dtypes[InputTokenNode] = token_sd
    node_shape_dtypes[DiffusionTokenNode] = token_sd

    gibbs_spec = BlockGibbsSpec(
        free_super_blocks=free_super_blocks,
        clamped_blocks=[input_block, terminal_block],
        node_shape_dtypes=node_shape_dtypes,
    )

    interaction_groups: List[InteractionGroup] = []
    for idx, block in enumerate(latent_blocks):
        next_block = terminal_block if idx == n_steps - 1 else latent_blocks[idx + 1]
        step_info = jnp.full((len(block),), idx, dtype=jnp.int32)
        interaction_groups.append(InteractionGroup(step_info, block, [next_block, input_block]))

    samplers: List[TextDiffusionSampler] = []
    for block in gibbs_spec.free_blocks:
        node = block.nodes[0]
        if not isinstance(node, DiffusionTokenNode):
            raise RuntimeError("Expected diffusion nodes in free blocks")
        samplers.append(TextDiffusionSampler(model=model, step_index=node.step))

    program = BlockSamplingProgram(gibbs_spec, samplers, interaction_groups)

    block_indices = {id(block): idx for idx, block in enumerate(gibbs_spec.blocks)}
    latent_block_indices = {}
    for block in latent_blocks:
        node = block.nodes[0]
        latent_block_indices[node.step] = block_indices[id(block)]

    input_block_index = block_indices[id(input_block)]
    terminal_block_index = block_indices[id(terminal_block)]

    return TextDiffusionProgramSpec(
        program=program,
        gibbs_spec=gibbs_spec,
        input_block=input_block,
        latent_blocks=latent_blocks,
        terminal_block=terminal_block,
        latent_block_indices=latent_block_indices,
        input_block_index=input_block_index,
        terminal_block_index=terminal_block_index,
    )
