import collections
import os
from typing import Optional, Dict, Union, Any, List

import torch
import wandb
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from src.utils.fileIO.model import save_model
from src.utils.forTrain.OOM import afterOOM
from src.utils.result.metric_io import save_metric


def upload_model_to_wandb(model_output_dir, name, metadata=None):
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
    artifact.add_dir(model_output_dir)
    wandb.run.log_artifact(artifact)


class CTCTrainer(Trainer):
    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None,
                 lr_warmup_ratio=0.1, lr_constant_ratio=0.2, sampling_rate=16_000, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        # self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer) -> None:

        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (
                    warmup_steps + constant_steps):
                return 1
            else:
                return max(
                    0.0, float(num_training_steps - current_step) / float(
                        max(1, num_training_steps - (warmup_steps + constant_steps)))
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""

        # adding an extra dimmention for the channels as our KeSpeech is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs = inputs["input_values"].unsqueeze(1)

        transformed_inputs = self.augmentator(transformed_inputs, sample_rate=self.sampling_rate)

        # returning the inputs to the original shape
        transformed_inputs = torch.squeeze(transformed_inputs, 1)

        inputs["input_values"] = transformed_inputs

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.empty_cache()
        model.train()
        inputs = self._prepare_inputs(inputs)
        if inputs["attention_mask"] is None:
            raise ValueError(
                "You must provide attention mask while training the model"
            )
        try:
            out = model.forward(input_values=inputs["input_values"],
                                attention_mask=inputs["attention_mask"],
                                labels=inputs["labels"])
        except torch.cuda.OutOfMemoryError as error:
            del inputs
            afterOOM(self.args.output_dir, error)
        del inputs
        loss = out["loss"]["total"]
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError as error:
            afterOOM(self.args.output_dir, error)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        if self.upload_model_to_wandb_each_step is not None and self.state.global_step > 0 \
                and self.state.global_step % self.upload_model_to_wandb_each_step == 0:
            upload_model_to_wandb(self.model_output_dir, name=f"{wandb.run.name}_{self.state.global_step}",
                                  metadata={"loss": float(loss)})
        if self.args.save_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0:
            save_model(model, os.path.join(self.args.output_dir, "model.pt"), save_state_dict=True)
            print(f"pytorch model state_dict saved to {os.path.join(self.args.output_dir, 'model.pt')}")
            save_metric("loss", float(loss), save_dir=self.args.output_dir)
        return loss

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,
                        ):
        model.eval()
        # 制作输入
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                out = model.forward(input_values=inputs["input_values"],
                                    attention_mask=inputs["attention_mask"],
                                    labels=inputs["labels"])
                loss = out["loss"]["total"].detach().cpu()
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                outputs = out["pred"]

            logits = outputs["asr"]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits).cpu()
        return loss, logits, inputs["labels"].cpu()


class FuseTrainer(Trainer):

    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None,
                 lr_warmup_ratio=0.1,
                 lr_constant_ratio=0.2, sampling_rate=16_000, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        # self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer) -> None:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        This method was built based on https://arxiv.org/pdf/2006.13979 :
            "The learning rate schedule has three phases: warm up for the first 10% of updates,
             keep constant for 40% and then linearly decay for the remainder"

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (
                    warmup_steps + constant_steps):
                return 1
            else:
                return max(
                    0.0, float(num_training_steps - current_step) / float(
                        max(1, num_training_steps - (warmup_steps + constant_steps)))
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""

        # adding an extra dimmention for the channels as our KeSpeech is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs = inputs["input_values"].unsqueeze(1)

        transformed_inputs = self.augmentator(transformed_inputs, sample_rate=self.sampling_rate)

        # returning the inputs to the original shape
        transformed_inputs = torch.squeeze(transformed_inputs, 1)

        inputs["input_values"] = transformed_inputs

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model.train()
        inputs = self._prepare_inputs(inputs)
        if inputs["attention_mask"] is None:
            raise ValueError(
                "You must provide attention mask while training the model"
            )
        try:
            out = model.forward(input_values=inputs["input_values"],
                                attention_mask=inputs["attention_mask"],
                                need_mse=True)  # 不使用label
        except torch.cuda.OutOfMemoryError as error:
            del inputs
            afterOOM(self.args.output_dir, error)
        del inputs
        loss = out["loss"]["fuse"]
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError as error:
            afterOOM(self.args.output_dir, error)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        if self.upload_model_to_wandb_each_step is not None and self.state.global_step > 0 \
                and self.state.global_step % self.upload_model_to_wandb_each_step == 0:
            upload_model_to_wandb(self.model_output_dir, name=f"{wandb.run.name}_{self.state.global_step}",
                                  metadata={"loss": float(loss)})
        if self.args.save_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0:
            save_model(model, os.path.join(self.args.output_dir, "model.pt"), save_state_dict=True)
            print(f"pytorch model state_dict saved to {os.path.join(self.args.output_dir, 'model.pt')}")
            save_metric("loss", float(loss), save_dir=self.args.output_dir)
        return loss

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,
                        ):
        # 制作输入
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                out = model.forward(input_values=inputs["input_values"],
                                    attention_mask=inputs["attention_mask"],
                                    labels=inputs["labels"], need_mse=True)

                loss = out["loss"]["fuse"].detach().cpu()
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                outputs = out["pred"]
            logits = outputs["asr"]
            del outputs, out
        if prediction_loss_only:
            return loss, None, None
        logits = nested_detach(logits).cpu()
        labels = inputs["labels"].cpu()
        del inputs
        return loss, logits, (labels)


class myModelTrainer(Trainer):

    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None,
                 lr_warmup_ratio=0.1,
                 lr_constant_ratio=0.2, sampling_rate=16_000, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        # self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer) -> None:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        This method was built based on https://arxiv.org/pdf/2006.13979 :
            "The learning rate schedule has three phases: warm up for the first 10% of updates,
             keep constant for 40% and then linearly decay for the remainder"

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (
                    warmup_steps + constant_steps):
                return 1
            else:
                return max(
                    0.0, float(num_training_steps - current_step) / float(
                        max(1, num_training_steps - (warmup_steps + constant_steps)))
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""

        # adding an extra dimmention for the channels as our KeSpeech is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs = inputs["input_values"].unsqueeze(1)

        transformed_inputs = self.augmentator(transformed_inputs, sample_rate=self.sampling_rate)

        # returning the inputs to the original shape
        transformed_inputs = torch.squeeze(transformed_inputs, 1)

        inputs["input_values"] = transformed_inputs

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model.train()
        inputs = self._prepare_inputs(inputs)
        if inputs["attention_mask"] is None:
            raise ValueError(
                "You must provide attention mask while training the model"
            )
        try:
            out = model.forward(input_values=inputs["input_values"],
                                attention_mask=inputs["attention_mask"],
                                labels=inputs["labels"])
        except torch.cuda.OutOfMemoryError as error:
            del inputs
            afterOOM(self.args.output_dir, error)
        del inputs
        loss = out["loss"]["asr"]
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        if self.upload_model_to_wandb_each_step is not None and self.state.global_step > 0 \
                and self.state.global_step % self.upload_model_to_wandb_each_step == 0:
            upload_model_to_wandb(self.model_output_dir, name=f"{wandb.run.name}_{self.state.global_step}",
                                  metadata={"loss": float(loss)})
        if self.args.save_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0:
            save_model(model, os.path.join(self.args.output_dir, "model.pt"), save_state_dict=True)
            print(f"pytorch model state_dict saved to {os.path.join(self.args.output_dir, 'model.pt')}")
            save_metric("loss", float(loss), save_dir=self.args.output_dir)
        return loss

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,
                        ):
        # 制作输入
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                out = model.forward(input_values=inputs["input_values"],
                                    attention_mask=inputs["attention_mask"],
                                    labels=inputs["labels"])

                loss = out["loss"]["asr"].detach().cpu()
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                outputs = out["pred"]
            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
        if prediction_loss_only:
            return loss, None, None
        logits = nested_detach(logits).cpu()
        return loss, logits, (inputs["labels"].cpu())


class AIDTrainer(Trainer):
    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None,
                 lr_warmup_ratio=0.1,
                 lr_constant_ratio=0.4, sampling_rate=16_000, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        # self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer) -> None:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        This method was built based on https://arxiv.org/pdf/2006.13979 :
            "The learning rate schedule has three phases: warm up for the first 10% of updates,
             keep constant for 40% and then linearly decay for the remainder"

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (
                    warmup_steps + constant_steps):
                return 1
            else:
                return max(
                    0.0, float(num_training_steps - current_step) / float(
                        max(1, num_training_steps - (warmup_steps + constant_steps)))
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""

        # adding an extra dimmention for the channels as our KeSpeech is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs = inputs["input_values"].unsqueeze(1)

        transformed_inputs = self.augmentator(transformed_inputs, sample_rate=self.sampling_rate)

        # returning the inputs to the original shape
        transformed_inputs = torch.squeeze(transformed_inputs, 1)

        inputs["input_values"] = transformed_inputs

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model.train()
        inputs = self._prepare_inputs(inputs)
        if inputs["attention_mask"] is None:
            raise ValueError(
                "You must provide attention mask while training the model"
            )
        try:
            out = model.forward(input_values=inputs["input_values"],
                                attention_mask=inputs["attention_mask"],
                                class_labels=inputs["dialect_index"])
        except torch.cuda.OutOfMemoryError as error:
            del inputs
            afterOOM(self.args.output_dir, error)
        del inputs
        loss = out["loss"]["total"]
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        if self.upload_model_to_wandb_each_step is not None and self.state.global_step > 0 \
                and self.state.global_step % self.upload_model_to_wandb_each_step == 0:
            upload_model_to_wandb(self.model_output_dir, name=f"{wandb.run.name}_{self.state.global_step}",
                                  metadata={"loss": float(loss)})
        if self.args.save_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0:
            save_model(model, os.path.join(self.args.output_dir, "model.pt"), save_state_dict=True)
            print(f"pytorch model state_dict saved to {os.path.join(self.args.output_dir, 'model.pt')}")
            save_metric("loss", float(loss), save_dir=self.args.output_dir)
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        # 制作输入
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                out = model.forward(input_values=inputs["input_values"],
                                    attention_mask=inputs["attention_mask"],
                                    class_labels=inputs["dialect_index"])
                loss = out["loss"]["total"].detach().cpu()
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                outputs = out["pred"]
            logits = outputs["aid"]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits).cpu()
        # if len(logits) == 1:
        #     logits = logits[0]

        return loss, logits, inputs["dialect_index"].cpu()


class Wa2vec2TripletTrainer(Trainer):
    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None,
                 lr_warmup_ratio=0.2, lr_constant_ratio=0.2, sampling_rate=16_000, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        # self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate
        self.alpha = alpha
        self.beta = beta

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer) -> None:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        This method was built based on https://arxiv.org/pdf/2006.13979 :
            "The learning rate schedule has three phases: warm up for the first 10% of updates,
             keep constant for 40% and then linearly decay for the remainder"

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (
                    warmup_steps + constant_steps):
                return 1
            else:
                return max(
                    0.0, float(num_training_steps - current_step) / float(
                        max(1, num_training_steps - (warmup_steps + constant_steps)))
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""

        # adding an extra dimmention for the channels as our KeSpeech is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs_P = inputs["input_values"]["P"].unsqueeze(1)
        transformed_inputs_A = inputs["input_values"]["A"].unsqueeze(1)
        transformed_inputs_N = inputs["input_values"]["N"].unsqueeze(1)

        transformed_inputs_P = self.augmentator(transformed_inputs_P, sample_rate=self.sampling_rate)
        transformed_inputs_A = self.augmentator(transformed_inputs_A, sample_rate=self.sampling_rate)
        transformed_inputs_N = self.augmentator(transformed_inputs_N, sample_rate=self.sampling_rate)

        # returning the inputs to the original shape
        transformed_inputs_P = torch.squeeze(transformed_inputs_P, 1)
        transformed_inputs_A = torch.squeeze(transformed_inputs_A, 1)
        transformed_inputs_N = torch.squeeze(transformed_inputs_N, 1)

        inputs["input_values"]["P"] = transformed_inputs_P
        inputs["input_values"]["A"] = transformed_inputs_A
        inputs["input_values"]["N"] = transformed_inputs_N

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model.train()
        inputs = self._prepare_inputs(inputs)

        if inputs["attention_mask"]["P"] is None or inputs["attention_mask"]["A"] is None or inputs["attention_mask"][
            "N"] is None:
            raise ValueError(
                "You must provide attention mask while training the model"
            )
        try:
            all_loss = model.forward(wavP=inputs["input_values"]["P"],
                                     wavA=inputs["input_values"]["A"],
                                     wavN=inputs["input_values"]["N"],
                                     pad_maskP=inputs["attention_mask"]["P"],
                                     pad_maskA=inputs["attention_mask"]["A"],
                                     pad_maskN=inputs["attention_mask"]["N"],
                                     labelP=inputs["labelAP"],
                                     labelN=inputs["labelN"],
                                     alpha=self.alpha,
                                     beta=self.beta)["loss"]
        except torch.cuda.OutOfMemoryError as error:
            del inputs
            afterOOM(self.args.output_dir, error)
        del inputs
        loss = all_loss["total"]
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        if self.upload_model_to_wandb_each_step is not None and self.state.global_step > 0 \
                and self.state.global_step % self.upload_model_to_wandb_each_step == 0:
            upload_model_to_wandb(self.model_output_dir, name=f"{wandb.run.name}_{self.state.global_step}",
                                  metadata={"loss": float(loss)})
        if self.args.save_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0:
            save_model(model, os.path.join(self.args.output_dir, "model.pt"), save_state_dict=True)

            print(f"pytorch model state_dict saved to {os.path.join(self.args.output_dir, 'model.pt')}")
            save_metric("loss", float(loss), save_dir=self.args.output_dir)
        if self.args.logging_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            print(f"\n ce_loss:{all_loss['aid']}, triplet_loss:{all_loss['triplet']}, total_loss:{loss}")
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        model.eval()
        # 制作输入
        inputs = self._prepare_inputs(inputs)
        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                out = model.forward(wavP=inputs["input_values"]["P"],
                                    wavA=inputs["input_values"]["A"],
                                    wavN=inputs["input_values"]["N"],
                                    pad_maskP=inputs["attention_mask"]["P"],
                                    pad_maskA=inputs["attention_mask"]["A"],
                                    pad_maskN=inputs["attention_mask"]["N"],
                                    labelP=inputs["labelAP"],
                                    labelN=inputs["labelN"])
                loss = out["loss"]["total"].detach().cpu()
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                outputs = out["pred"]
            logitsA = outputs["predA"]
            logitsP = outputs["predP"]
            logitsN = outputs["predN"]
        if prediction_loss_only:
            return (loss, None, None)
        logitsA = nested_detach(logitsA)
        logitsP = nested_detach(logitsP)
        logitsN = nested_detach(logitsN)
        logits = {"logitsA": logitsA, "logitsP": logitsP, "logitsN": logitsN}
        labels = {"labelA": inputs["labelAP"].cpu(), "labelP": inputs["labelAP"].cpu(),
                  "labelN": inputs["labelN"].cpu()}
        return loss, logits, labels


class ECAPATDNNTripletTrainer(Trainer):
    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None,
                 lr_warmup_ratio=0.1, lr_constant_ratio=0.2, sampling_rate=16_000, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        # self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate
        self.alpha = alpha
        self.beta = beta

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer) -> None:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        This method was built based on https://arxiv.org/pdf/2006.13979 :
            "The learning rate schedule has three phases: warm up for the first 10% of updates,
             keep constant for 40% and then linearly decay for the remainder"

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (
                    warmup_steps + constant_steps):
                return 1
            else:
                return max(
                    0.0, float(num_training_steps - current_step) / float(
                        max(1, num_training_steps - (warmup_steps + constant_steps)))
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""

        # adding an extra dimmention for the channels as our KeSpeech is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs_P = inputs["input_values"]["P"].unsqueeze(1)
        transformed_inputs_A = inputs["input_values"]["A"].unsqueeze(1)
        transformed_inputs_N = inputs["input_values"]["N"].unsqueeze(1)

        transformed_inputs_P = self.augmentator(transformed_inputs_P, sample_rate=self.sampling_rate)
        transformed_inputs_A = self.augmentator(transformed_inputs_A, sample_rate=self.sampling_rate)
        transformed_inputs_N = self.augmentator(transformed_inputs_N, sample_rate=self.sampling_rate)

        # returning the inputs to the original shape
        transformed_inputs_P = torch.squeeze(transformed_inputs_P, 1)
        transformed_inputs_A = torch.squeeze(transformed_inputs_A, 1)
        transformed_inputs_N = torch.squeeze(transformed_inputs_N, 1)

        inputs["input_values"]["P"] = transformed_inputs_P
        inputs["input_values"]["A"] = transformed_inputs_A
        inputs["input_values"]["N"] = transformed_inputs_N

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model.train()
        inputs = self._prepare_inputs(inputs)

        if inputs["lengths"]["P"] is None or inputs["lengths"]["A"] is None or inputs["lengths"][
            "N"] is None:
            raise ValueError(
                "You must provide lengths while training the model"
            )
        try:
            all_loss = model.forward(wavP=inputs["input_values"]["P"],
                                     wavA=inputs["input_values"]["A"],
                                     wavN=inputs["input_values"]["N"],
                                     lens_P=inputs["lengths"]["P"],
                                     lens_A=inputs["lengths"]["A"],
                                     lens_N=inputs["lengths"]["N"],
                                     labelP=inputs["labelAP"],
                                     labelN=inputs["labelN"],
                                     alpha=self.alpha,
                                     beta=self.beta)["loss"]
        except torch.cuda.OutOfMemoryError as error:
            del inputs
            afterOOM(self.args.output_dir, error)
        del inputs
        loss = all_loss["total"]
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        if self.upload_model_to_wandb_each_step is not None and self.state.global_step > 0 \
                and self.state.global_step % self.upload_model_to_wandb_each_step == 0:
            upload_model_to_wandb(self.model_output_dir, name=f"{wandb.run.name}_{self.state.global_step}",
                                  metadata={"loss": float(loss)})
        if self.args.save_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0:
            save_model(model, os.path.join(self.args.output_dir, "model.pt"), save_state_dict=True)

            print(f"pytorch model state_dict saved to {os.path.join(self.args.output_dir, 'model.pt')}")
            save_metric("loss", float(loss), save_dir=self.args.output_dir)
        if self.args.logging_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            print(f"\n ce_loss:{all_loss['aid']}, triplet_loss:{all_loss['triplet']}, total_loss:{loss}")
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        model.eval()
        # 制作输入
        inputs = self._prepare_inputs(inputs)
        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                out = model.forward(wavP=inputs["input_values"]["P"],
                                    wavA=inputs["input_values"]["A"],
                                    wavN=inputs["input_values"]["N"],
                                    lens_P=inputs["lengths"]["P"],
                                    lens_A=inputs["lengths"]["A"],
                                    lens_N=inputs["lengths"]["N"],
                                    labelP=inputs["labelAP"],
                                    labelN=inputs["labelN"])
                loss = out["loss"]["total"].detach().cpu()
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                outputs = out["pred"]
            logitsA = outputs["predA"]
            logitsP = outputs["predP"]
            logitsN = outputs["predN"]
        if prediction_loss_only:
            return (loss, None, None)
        logitsA = nested_detach(logitsA)
        logitsP = nested_detach(logitsP)
        logitsN = nested_detach(logitsN)
        logits = {"logitsA": logitsA, "logitsP": logitsP, "logitsN": logitsN}
        labels = {"labelA": inputs["labelAP"].cpu(), "labelP": inputs["labelAP"].cpu(),
                  "labelN": inputs["labelN"].cpu()}
        return loss, logits, labels


class GetFeatureTrainer(Trainer):
    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None,
                 lr_warmup_ratio=0.1, lr_constant_ratio=0.2, sampling_rate=16_000, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        # self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer) -> None:

        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (
                    warmup_steps + constant_steps):
                return 1
            else:
                return max(
                    0.0, float(num_training_steps - current_step) / float(
                        max(1, num_training_steps - (warmup_steps + constant_steps)))
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""

        # adding an extra dimmention for the channels as our KeSpeech is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs = inputs["input_values"].unsqueeze(1)

        transformed_inputs = self.augmentator(transformed_inputs, sample_rate=self.sampling_rate)

        # returning the inputs to the original shape
        transformed_inputs = torch.squeeze(transformed_inputs, 1)

        inputs["input_values"] = transformed_inputs

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        pass

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,
                        ):
        model.eval()
        # 制作输入
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        with torch.no_grad():
            out = model.encode_batch(wavs=inputs["input_values"], wav_lens=inputs["lengths"])

        logits = nested_detach(out).cpu()
        return torch.tensor(0.0), logits, logits


class JMAATrainer(Trainer):
    def __init__(self, model_output_dir, length_field_name="length", upload_model_to_wandb_each_step=None,
                 lr_warmup_ratio=0.1, lr_constant_ratio=0.2, sampling_rate=16_000, **kwargs):
        super().__init__(**kwargs)
        self.model_output_dir = model_output_dir
        # self.length_field_name = length_field_name
        self.upload_model_to_wandb_each_step = upload_model_to_wandb_each_step
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_constant_ratio = lr_constant_ratio
        self.sampling_rate = sampling_rate

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps: int, optimizer) -> None:

        def lr_lambda(current_step):
            warmup_steps = int(num_training_steps * self.lr_warmup_ratio)
            constant_steps = int(num_training_steps * self.lr_constant_ratio)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            elif (self.lr_warmup_ratio + self.lr_constant_ratio) == 1.0 or current_step < (
                    warmup_steps + constant_steps):
                return 1
            else:
                return max(
                    0.0, float(num_training_steps - current_step) / float(
                        max(1, num_training_steps - (warmup_steps + constant_steps)))
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _apply_some_audio_transformations(self, inputs):
        """Perform some audio transformations"""

        # adding an extra dimmention for the channels as our KeSpeech is mono audio and
        # the expected shape of input for torch_audiomentations is (batch_size, num_channels, num_samples)
        transformed_inputs = inputs["input_values"].unsqueeze(1)

        transformed_inputs = self.augmentator(transformed_inputs, sample_rate=self.sampling_rate)

        # returning the inputs to the original shape
        transformed_inputs = torch.squeeze(transformed_inputs, 1)

        inputs["input_values"] = transformed_inputs

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        torch.cuda.empty_cache()
        model.train()
        inputs = self._prepare_inputs(inputs)
        if inputs["attention_mask"] is None:
            raise ValueError(
                "You must provide attention mask while training the model"
            )
        try:
            out = model.forward(input_values=inputs["input_values"],
                                attention_mask=inputs["attention_mask"],
                                asr_labels=inputs["labels"],
                                aid_labels=inputs["dialect_index"])
        except torch.cuda.OutOfMemoryError as error:
            del inputs
            afterOOM(self.args.output_dir, error)
        del inputs
        loss = out["loss"]["total"]
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        try:
            loss.backward()
        except torch.cuda.OutOfMemoryError as error:
            afterOOM(self.args.output_dir, error)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        if self.upload_model_to_wandb_each_step is not None and self.state.global_step > 0 \
                and self.state.global_step % self.upload_model_to_wandb_each_step == 0:
            upload_model_to_wandb(self.model_output_dir, name=f"{wandb.run.name}_{self.state.global_step}",
                                  metadata={"loss": float(loss)})
        if self.args.save_steps is not None and self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0:
            save_model(model, os.path.join(self.args.output_dir, "model.pt"), save_state_dict=True)
            print(f"pytorch model state_dict saved to {os.path.join(self.args.output_dir, 'model.pt')}")
            save_metric("loss", float(loss), save_dir=self.args.output_dir)
        return loss

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,
                        ):
        model.eval()
        # 制作输入
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                out = model.forward(input_values=inputs["input_values"],
                                    attention_mask=inputs["attention_mask"],
                                    asr_labels=inputs["labels"],
                                    aid_labels=inputs["dialect_index"])
                loss = out["loss"]["total"].detach().cpu()
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                outputs = out["pred"]

            logits = outputs["asr"]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits).cpu()
        return loss, logits, inputs["labels"].cpu()
