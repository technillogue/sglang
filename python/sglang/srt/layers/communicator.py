# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This module handles communication patterns for distributed training in the SGLang framework.
It defines how data is scattered or gathered across different ranks in a distributed setup,
particularly for attention and MLP layers in transformer models.
"""

from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Dict, Optional

import torch.distributed

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    attn_tp_reduce_scatter,
    dp_gather_partial,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class DataDistributionMode(Enum):
    """
    Defines the data distribution modes across ranks in a distributed training setup.
    - DISTRIBUTED: Data is split across all ranks, each rank has a unique portion.
    - TP_GROUP_FULL: Data is fully replicated within each tensor parallel attention group.
    - GLOBAL_FULL: Data is fully replicated across all ranks.
    
    Example:
    Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles sequences a,b,c,d
    Model input/output: [ab, ab, cd, cd] for four ranks respectively
    - DISTRIBUTED: [a, b, c, d]
    - TP_GROUP_FULL: [ab, ab, cd, cd], i.e., all ranks inside a TP attention group have full data of the group
    - GLOBAL_FULL: [abcd, abcd, abcd, abcd]
    """

    DISTRIBUTED = auto()
    TP_GROUP_FULL = auto()
    GLOBAL_FULL = auto()

    @staticmethod
    def default_model_io_mode():
        """Returns the distribution mode used for model forward pass input and output data."""
        return DataDistributionMode.TP_GROUP_FULL


@dataclass
class LayerContext:
    """Context for computing distribution modes for different layers in the model."""
    total_layers: int
    current_layer_id: int
    is_current_layer_sparse: bool
    is_previous_layer_sparse: Optional[bool]

    def get_previous_layer_context(self):
        """Returns the context for the previous layer."""
        assert self.is_previous_layer_sparse is not None
        return LayerContext(
            current_layer_id=self.current_layer_id - 1,
            is_current_layer_sparse=self.is_previous_layer_sparse,
            is_previous_layer_sparse=None,
            total_layers=self.total_layers,
        )


@dataclass
class LayerDistributionConfig:
    """
    Defines the distribution modes for different components of a transformer layer.
    This helps in determining how data is distributed or gathered at each stage of the layer computation.
    """
    input_distribution: DataDistributionMode
    attention_distribution: DataDistributionMode
    mlp_distribution: DataDistributionMode
    intermediate_residual_distribution: DataDistributionMode
    output_distribution: DataDistributionMode

    @classmethod
    def create(cls, **kwargs):
        """Initializes distribution modes for a layer based on the provided context."""
        context = LayerContext(**kwargs)
        return cls(
            input_distribution=cls._determine_input_distribution(context),
            attention_distribution=DataDistributionMode.TP_GROUP_FULL,
            mlp_distribution=cls._determine_mlp_distribution(context),
            intermediate_residual_distribution=cls._determine_intermediate_residual_distribution(context),
            output_distribution=cls._determine_output_distribution(context),
        )

    @classmethod
    def _determine_input_distribution(cls, context: LayerContext):
        """Computes the input distribution mode for the current layer."""
        if context.current_layer_id == 0:
            return DataDistributionMode.default_model_io_mode()
        return cls._determine_output_distribution(context.get_previous_layer_context())

    @classmethod
    def _determine_mlp_distribution(cls, context: LayerContext):
        """Computes the distribution mode for the MLP component of the layer."""
        if context.is_current_layer_sparse:
            return (
                DataDistributionMode.DISTRIBUTED
                if global_server_args_dict["enable_deepep_moe"]
                else DataDistributionMode.GLOBAL_FULL
            )
        else:
            return (
                DataDistributionMode.DISTRIBUTED
                if is_moe_dense_fully_distributed()
                else DataDistributionMode.GLOBAL_FULL
            )

    @classmethod
    def _determine_intermediate_residual_distribution(cls, context: LayerContext):
        """Computes the distribution mode for the intermediate residual connection."""
        mlp_dist = cls._determine_mlp_distribution(context)
        if mlp_dist == DataDistributionMode.DISTRIBUTED:
            return DataDistributionMode.DISTRIBUTED
        if mlp_dist == DataDistributionMode.GLOBAL_FULL:
            return DataDistributionMode.TP_GROUP_FULL
        raise NotImplementedError

    @classmethod
    def _determine_output_distribution(cls, context: LayerContext):
        """Computes the output distribution mode for the current layer."""
        mlp_dist = cls._determine_mlp_distribution(context)
        if context.current_layer_id == context.total_layers - 1:
            return DataDistributionMode.default_model_io_mode()
        if mlp_dist == DataDistributionMode.DISTRIBUTED:
            return DataDistributionMode.DISTRIBUTED
        if mlp_dist == DataDistributionMode.GLOBAL_FULL:
            return DataDistributionMode.TP_GROUP_FULL
        raise NotImplementedError


def is_moe_dense_fully_distributed():
    """Checks if MoE dense layers should use fully distributed data parallelism."""
    return global_server_args_dict["moe_dense_tp_size"] == 1


class LayerDataCommunicator:
    """
    Manages communication patterns for a transformer layer, handling data distribution
    and gathering for attention and MLP components.
    """
    def __init__(
        self,
        distribution_config: LayerDistributionConfig,
        input_normalization: torch.nn.Module,
        post_attention_normalization: torch.nn.Module,
    ):
        self.distribution_config = distribution_config
        self.input_normalization = input_normalization
        self.post_attention_normalization = post_attention_normalization

        self._comm_context = CommunicationContext.initialize()
        self._basic_communication_handler = BasicCommunicationHandler.get_handler(
            source_mode=self.distribution_config.input_distribution,
            target_mode=self.distribution_config.attention_distribution,
            context=self._comm_context,
        )
        self._attention_to_mlp_communication_handler = AttentionToMlpCommunicationHandler.get_handler(
            attention_data_mode=self.distribution_config.attention_distribution,
            residual_input_mode=self.distribution_config.input_distribution,
            mlp_data_mode=self.distribution_config.mlp_distribution,
            residual_output_mode=self.distribution_config.intermediate_residual_distribution,
            context=self._comm_context,
        )
        self._layer_output_communication_handler = LayerOutputCommunicationHandler.get_handler(
            mlp_data_mode=self.distribution_config.mlp_distribution,
            residual_input_mode=self.distribution_config.intermediate_residual_distribution,
            target_mode=self.distribution_config.output_distribution,
            context=self._comm_context,
        )

    def prepare_for_attention(
        self,
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
    ):
        """
        Prepares data for the attention computation by applying layer normalization
        and necessary communication patterns.
        """
        if hidden_data.shape[0] == 0:
            residual_data = hidden_data
        else:
            if residual_data is None:
                residual_data = hidden_data
                hidden_data = self.input_normalization(hidden_data)
            else:
                hidden_data, residual_data = self.input_normalization(hidden_data, residual_data)

        hidden_data = self._basic_communication_handler(
            hidden_data=hidden_data,
            batch_info=batch_info,
            context=self._comm_context,
        )

        return hidden_data, residual_data

    def prepare_for_mlp(
        self,
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
    ):
        """
        Prepares data for the MLP computation by applying communication patterns
        and layer normalization.
        """
        return self._attention_to_mlp_communication_handler(
            hidden_data=hidden_data,
            residual_data=residual_data,
            batch_info=batch_info,
            normalization_layer=self.post_attention_normalization,
            context=self._comm_context,
        )

    def finalize_layer_output(
        self,
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
    ):
        """
        Post-processes the layer output by applying necessary communication patterns
        to combine hidden states and residuals.
        """
        return self._layer_output_communication_handler(
            hidden_data=hidden_data,
            residual_data=residual_data,
            batch_info=batch_info,
            context=self._comm_context,
        )


@dataclass
class CommunicationContext:
    """
    Holds configuration and context information for communication operations
    in distributed training.
    """
    group_sizes: Dict[DataDistributionMode, int]
    attention_tp_rank: int
    attention_tp_group_size: int
    attention_dp_group_size: int
    tensor_parallel_size: int

    def has_same_group_size(self, mode_a: DataDistributionMode, mode_b: DataDistributionMode):
        """Checks if two distribution modes have the same process group size."""
        return self.group_sizes[mode_a] == self.group_sizes[mode_b]

    @classmethod
    def initialize(cls):
        """Initializes a new communication context with settings from the distributed environment."""
        attention_tp_rank = get_attention_tp_rank()
        attention_tp_group_size = get_attention_tp_size()
        attention_dp_group_size = get_attention_dp_size()
        tensor_parallel_size = get_tensor_model_parallel_world_size()
        group_sizes = {
            DataDistributionMode.DISTRIBUTED: 1,
            DataDistributionMode.TP_GROUP_FULL: attention_tp_group_size,
            # TODO: support --moe-dense-tp-size > 1
            DataDistributionMode.GLOBAL_FULL: tensor_parallel_size,
        }
        return cls(
            group_sizes=group_sizes,
            attention_tp_rank=attention_tp_rank,
            attention_tp_group_size=attention_tp_group_size,
            attention_dp_group_size=attention_dp_group_size,
            tensor_parallel_size=tensor_parallel_size,
        )


class BasicCommunicationHandler:
    """Handles basic communication operations between different distribution modes."""

    @staticmethod
    def get_handler(
        source_mode: DataDistributionMode,
        target_mode: DataDistributionMode,
        context: CommunicationContext,
    ):
        """Selects the appropriate communication handler based on source and target distribution modes."""
        if context.has_same_group_size(source_mode, target_mode):
            return BasicCommunicationHandler._no_op_handler

        if (source_mode == DataDistributionMode.DISTRIBUTED) and (
            target_mode == DataDistributionMode.TP_GROUP_FULL
        ):
            return BasicCommunicationHandler._distributed_to_tp_group_full

        raise NotImplementedError(f"{source_mode=} {target_mode=}")

    @staticmethod
    def _no_op_handler(
        hidden_data: torch.Tensor,
        batch_info: ForwardBatch,
        context: CommunicationContext,
    ) -> torch.Tensor:
        """Returns the input tensor unchanged if no communication is needed."""
        return hidden_data

    @staticmethod
    def _distributed_to_tp_group_full(
        hidden_data: torch.Tensor,
        batch_info: ForwardBatch,
        context: CommunicationContext,
    ) -> torch.Tensor:
        """
        Gathers distributed data into a full tensor within each tensor parallel attention group.
        """
        hidden_data, local_hidden_data = (
            batch_info.gathered_buffer[: batch_info.input_ids.shape[0]],
            hidden_data,
        )
        attn_tp_all_gather(
            list(hidden_data.tensor_split(context.attention_tp_group_size)),
            local_hidden_data,
        )
        return hidden_data


class AttentionToMlpCommunicationHandler:
    """
    Manages communication operations that involve all-reduce operations
    and layer normalization for transitioning from attention to MLP layers.
    """

    @staticmethod
    def get_handler(
        attention_data_mode: DataDistributionMode,
        residual_input_mode: DataDistributionMode,
        mlp_data_mode: DataDistributionMode,
        residual_output_mode: DataDistributionMode,
        context: CommunicationContext,
    ):
        """
        Selects the appropriate handler for communication and layer normalization
        based on input and output distribution modes.
        """
        if (
            context.has_same_group_size(
                attention_data_mode, mlp_data_mode
            )
            and context.has_same_group_size(residual_input_mode, residual_output_mode)
            and context.attention_tp_group_size == 1
        ):
            return AttentionToMlpCommunicationHandler._basic_normalization_handler

        if (
            (attention_data_mode == DataDistributionMode.TP_GROUP_FULL)
            and (
                residual_input_mode in [DataDistributionMode.DISTRIBUTED, DataDistributionMode.TP_GROUP_FULL]
            )
            and (mlp_data_mode == DataDistributionMode.GLOBAL_FULL)
            and (residual_output_mode == DataDistributionMode.TP_GROUP_FULL)
        ):
            return partial(
                AttentionToMlpCommunicationHandler._gather_attention_and_residual_data,
                residual_input_mode=residual_input_mode,
            )

        if (
            (attention_data_mode == DataDistributionMode.TP_GROUP_FULL)
            and (
                residual_input_mode in [DataDistributionMode.DISTRIBUTED, DataDistributionMode.TP_GROUP_FULL]
            )
            and (mlp_data_mode == DataDistributionMode.DISTRIBUTED)
            and (residual_output_mode == DataDistributionMode.DISTRIBUTED)
        ):
            return partial(
                AttentionToMlpCommunicationHandler._distribute_attention_and_residual_data,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{attention_data_mode=} {residual_input_mode=} {mlp_data_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _basic_normalization_handler(
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
        normalization_layer: torch.nn.Module,
        context: CommunicationContext,
    ):
        """Applies layer normalization without additional communication."""
        # TODO move these `if shape != 0` into LayerNorm itself
        if hidden_data.shape[0] != 0:
            hidden_data, residual_data = normalization_layer(hidden_data, residual_data)
        return hidden_data, residual_data

    @staticmethod
    def _gather_attention_and_residual_data(
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
        normalization_layer: torch.nn.Module,
        context: CommunicationContext,
        *,
        residual_input_mode,
    ):
        """
        Gathers hidden data and residuals across ranks, applies layer normalization,
        and handles data parallelism if enabled.
        """
        if residual_input_mode == DataDistributionMode.DISTRIBUTED:
            residual_data, local_residual_data = (
                batch_info.gathered_buffer[
                    : batch_info.input_ids.shape[0]
                ].clone(),
                residual_data,
            )
            attn_tp_all_gather(
                list(residual_data.tensor_split(context.attention_tp_group_size)), local_residual_data
            )
        if context.attention_dp_group_size != 1:
            if context.attention_tp_rank == 0:
                hidden_data += residual_data
            hidden_data, local_hidden_data = (
                batch_info.gathered_buffer,
                hidden_data,
            )
            dp_gather_partial(hidden_data, local_hidden_data, batch_info)
            dp_scatter(residual_data, hidden_data, batch_info)
            if hidden_data.shape[0] != 0:
                hidden_data = normalization_layer(hidden_data)
        else:
            hidden_data = tensor_model_parallel_all_reduce(hidden_data)
            hidden_data, residual_data = normalization_layer(hidden_data, residual_data)
        return hidden_data, residual_data

    @staticmethod
    def _distribute_attention_and_residual_data(
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
        normalization_layer: torch.nn.Module,
        context: CommunicationContext,
        *,
        residual_input_mode,
    ):
        """
        Distributes hidden data and residuals to individual ranks and applies layer normalization.
        """
        tensor_segments = list(hidden_data.tensor_split(context.attention_tp_group_size))
        hidden_data = tensor_segments[context.attention_tp_rank]
        attn_tp_reduce_scatter(hidden_data, tensor_segments)
        if residual_input_mode == DataDistributionMode.TP_GROUP_FULL:
            residual_data = residual_data.tensor_split(context.attention_tp_group_size)[context.attention_tp_rank]
        if hidden_data.shape[0] != 0:
            hidden_data, residual_data = normalization_layer(hidden_data, residual_data)
        return hidden_data, residual_data


class LayerOutputCommunicationHandler:
    """
    Handles communication for pairs of tensors (hidden data and residuals)
    that can be summed if needed during distributed processing at the layer output.
    """

    @classmethod
    def execute_handler(
        cls,
        hidden_data_mode,
        residual_input_mode,
        target_mode,
        context,
        **kwargs,
    ):
        """Executes the appropriate communication handler for tensor pairs."""
        return cls.get_handler(
            hidden_data_mode=hidden_data_mode,
            residual_input_mode=residual_input_mode,
            target_mode=target_mode,
            context=context,
        )(context=context, **kwargs)

    @staticmethod
    def get_handler(
        hidden_data_mode: DataDistributionMode,
        residual_input_mode: DataDistributionMode,
        target_mode: DataDistributionMode,
        context: CommunicationContext,
    ):
        """
        Selects the communication handler for tensor pairs based on input and output distribution modes.
        """
        if context.has_same_group_size(
            hidden_data_mode, target_mode
        ) and context.has_same_group_size(residual_input_mode, target_mode):
            return LayerOutputCommunicationHandler._no_op_handler

        if (
            (hidden_data_mode == DataDistributionMode.GLOBAL_FULL)
            and (residual_input_mode == DataDistributionMode.TP_GROUP_FULL)
            and (target_mode == DataDistributionMode.TP_GROUP_FULL)
        ):
            return LayerOutputCommunicationHandler._distribute_hidden_data

        if (
            (hidden_data_mode == DataDistributionMode.DISTRIBUTED)
            and (residual_input_mode == DataDistributionMode.DISTRIBUTED)
            and (target_mode == DataDistributionMode.TP_GROUP_FULL)
        ):
            return LayerOutputCommunicationHandler._gather_data

        if (
            (hidden_data_mode == DataDistributionMode.TP_GROUP_FULL)
            and (residual_input_mode == DataDistributionMode.TP_GROUP_FULL)
            and (target_mode == DataDistributionMode.DISTRIBUTED)
        ):
            return LayerOutputCommunicationHandler._distribute_data

        raise NotImplementedError(
            f"{hidden_data_mode=} {residual_input_mode=} {target_mode=}"
        )

    @staticmethod
    def _no_op_handler(
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
        context: CommunicationContext,
    ):
        """Returns the input tensors unchanged if no communication is needed."""
        return hidden_data, residual_data

    @staticmethod
    def _distribute_hidden_data(
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
        context: CommunicationContext,
    ):
        """
        Distributes hidden data across ranks while keeping residuals unchanged.
        Note: batch_info.gathered_buffer is used both after distribute and after gather.
        """
        # TODO(ch-wan): use reduce-scatter in MLP to avoid this scatter
        hidden_data, global_hidden_data = (
            batch_info.gathered_buffer[: batch_info.input_ids.shape[0]],
            hidden_data,
        )
        dp_scatter(hidden_data, global_hidden_data, batch_info)
        return hidden_data, residual_data

    @staticmethod
    def _gather_data(
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
        context: CommunicationContext,
    ):
        """
        Gathers distributed hidden data and residuals into a full tensor within
        each tensor parallel attention group.
        """
        hidden_data += residual_data
        residual_data = None
        hidden_data, local_hidden_data = (
            batch_info.gathered_buffer[: batch_info.input_ids.shape[0]],
            hidden_data,
        )
        attn_tp_all_gather(
            list(hidden_data.tensor_split(context.attention_tp_group_size)),
            local_hidden_data,
        )
        return hidden_data, residual_data

    @staticmethod
    def _distribute_data(
        hidden_data: torch.Tensor,
        residual_data: torch.Tensor,
        batch_info: ForwardBatch,
        context: CommunicationContext,
    ):
        """Distributes hidden data to individual ranks, assuming residual is None."""
        assert residual_data is None, "not yet handled residual_data!=None"
        tensor_segments = list(hidden_data.tensor_split(context.attention_tp_group_size))
        hidden_data = tensor_segments[context.attention_tp_rank]
        return hidden_data, residual_data
