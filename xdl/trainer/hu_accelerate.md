# Accelerator

The [Accelerator](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator) is the main class for enabling distributed training on any type of training setup. Read the [Add Accelerator to your code](../basic_tutorials/migration) tutorial to learn more about how to add the [Accelerator](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator) to your script.

## Accelerator[[api]][[accelerate.Accelerator]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class accelerate.Accelerator</name><anchor>accelerate.Accelerator</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L183</source><parameters>[{"name": "device_placement", "val": ": bool = True"}, {"name": "split_batches", "val": ": bool = <object object at 0x7f717339e710>"}, {"name": "mixed_precision", "val": ": PrecisionType | str | None = None"}, {"name": "gradient_accumulation_steps", "val": ": int = 1"}, {"name": "cpu", "val": ": bool = False"}, {"name": "dataloader_config", "val": ": DataLoaderConfiguration | None = None"}, {"name": "deepspeed_plugin", "val": ": DeepSpeedPlugin | dict[str, DeepSpeedPlugin] | None = None"}, {"name": "fsdp_plugin", "val": ": FullyShardedDataParallelPlugin | None = None"}, {"name": "torch_tp_plugin", "val": ": TorchTensorParallelPlugin | None = None"}, {"name": "megatron_lm_plugin", "val": ": MegatronLMPlugin | None = None"}, {"name": "rng_types", "val": ": list[str | RNGType] | None = None"}, {"name": "log_with", "val": ": str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None"}, {"name": "project_dir", "val": ": str | os.PathLike | None = None"}, {"name": "project_config", "val": ": ProjectConfiguration | None = None"}, {"name": "gradient_accumulation_plugin", "val": ": GradientAccumulationPlugin | None = None"}, {"name": "step_scheduler_with_optimizer", "val": ": bool = True"}, {"name": "kwargs_handlers", "val": ": list[KwargsHandler] | None = None"}, {"name": "dynamo_backend", "val": ": DynamoBackend | str | None = None"}, {"name": "dynamo_plugin", "val": ": TorchDynamoPlugin | None = None"}, {"name": "deepspeed_plugins", "val": ": DeepSpeedPlugin | dict[str, DeepSpeedPlugin] | None = None"}, {"name": "parallelism_config", "val": ": ParallelismConfig | None = None"}]</parameters><paramsdesc>- **device_placement** (`bool`, *optional*, defaults to `True`) --
  Whether or not the accelerator should put objects on device (tensors yielded by the dataloader, model,
  etc...).
- **mixed_precision** (`str`, *optional*) --
  Whether or not to use mixed precision training. Choose from 'no','fp16','bf16' or 'fp8'. Will default to
  the value in the environment variable `ACCELERATE_MIXED_PRECISION`, which will use the default value in the
  accelerate config of the current system or the flag passed with the `accelerate.launch` command. 'fp8'
  requires the installation of transformers-engine.
- **gradient_accumulation_steps** (`int`, *optional*, default to 1) --
  The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with
  `Accelerator.accumulate`. If not passed, will default to the value in the environment variable
  `ACCELERATE_GRADIENT_ACCUMULATION_STEPS`. Can also be configured through a `GradientAccumulationPlugin`.
- **cpu** (`bool`, *optional*) --
  Whether or not to force the script to execute on CPU. Will ignore GPU available if set to `True` and force
  the execution on one process only.
- **dataloader_config** (`DataLoaderConfiguration`, *optional*) --
  A configuration for how the dataloaders should be handled in distributed scenarios.
- **deepspeed_plugin** ([DeepSpeedPlugin](/docs/accelerate/v1.11.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin) or dict of `str` -- [DeepSpeedPlugin](/docs/accelerate/v1.11.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin), *optional*):
  Tweak your DeepSpeed related args using this argument. This argument is optional and can be configured
  directly using *accelerate config*. If using multiple plugins, use the configured `key` property of each
  plugin to access them from `accelerator.state.get_deepspeed_plugin(key)`. Alias for `deepspeed_plugins`.
- **fsdp_plugin** ([FullyShardedDataParallelPlugin](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.FullyShardedDataParallelPlugin), *optional*) --
  Tweak your FSDP related args using this argument. This argument is optional and can be configured directly
  using *accelerate config*
- **torch_tp_plugin** (`TorchTensorParallelPlugin`, *optional*) --
  Deprecated: use `parallelism_config` with `tp_size` instead.
- **megatron_lm_plugin** ([MegatronLMPlugin](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.MegatronLMPlugin), *optional*) --
  Tweak your MegatronLM related args using this argument. This argument is optional and can be configured
  directly using *accelerate config*
- **rng_types** (list of `str` or [RNGType](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.RNGType)) --
  The list of random number generators to synchronize at the beginning of each iteration in your prepared
  dataloaders. Should be one or several of:

  - `"torch"`: the base torch random number generator
  - `"cuda"`: the CUDA random number generator (GPU only)
  - `"xla"`: the XLA random number generator (TPU only)
  - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
    dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

  Will default to `["torch"]` for PyTorch versions <=1.5.1 and `["generator"]` for PyTorch versions >= 1.6.
- **log_with** (list of `str`, [LoggerType](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.LoggerType) or [GeneralTracker](/docs/accelerate/v1.11.0/en/package_reference/tracking#accelerate.tracking.GeneralTracker), *optional*) --
  A list of loggers to be setup for experiment tracking. Should be one or several of:

  - `"all"`
  - `"tensorboard"`
  - `"wandb"`
  - `"trackio"`
  - `"aim"`
  - `"comet_ml"`
  - `"mlflow"`
  - `"dvclive"`
  - `"swanlab"`
  If `"all"` is selected, will pick up all available trackers in the environment and initialize them. Can
  also accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
- **project_config** ([ProjectConfiguration](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.ProjectConfiguration), *optional*) --
  A configuration for how saving the state can be handled.
- **project_dir** (`str`, `os.PathLike`, *optional*) --
  A path to a directory for storing data such as logs of locally-compatible loggers and potentially saved
  checkpoints.
- **step_scheduler_with_optimizer** (`bool`, *optional*, defaults to `True`) --
  Set `True` if the learning rate scheduler is stepped at the same time as the optimizer, `False` if only
  done under certain circumstances (at the end of each epoch, for instance).
- **kwargs_handlers** (list of [KwargsHandler](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.KwargsHandler), *optional*) --
  A list of [KwargsHandler](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.KwargsHandler) to customize how the objects related to distributed training, profiling
  or mixed precision are created. See [kwargs](kwargs) for more information.
- **dynamo_backend** (`str` or [DynamoBackend](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.DynamoBackend), *optional*, defaults to `"no"`) --
  Set to one of the possible dynamo backends to optimize your training with torch dynamo.
- **dynamo_plugin** ([TorchDynamoPlugin](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.TorchDynamoPlugin), *optional*) --
  A configuration for how torch dynamo should be handled, if more tweaking than just the `backend` or `mode`
  is needed.
- **gradient_accumulation_plugin** ([GradientAccumulationPlugin](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.GradientAccumulationPlugin), *optional*) --
  A configuration for how gradient accumulation should be handled, if more tweaking than just the
  `gradient_accumulation_steps` is needed.</paramsdesc><paramgroups>0</paramgroups></docstring>

Creates an instance of an accelerator for distributed training or mixed precision training.



**Available attributes:**

- **device** (`torch.device`) -- The device to use.
- **distributed_type** ([DistributedType](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.DistributedType)) -- The distributed training configuration.
- **local_process_index** (`int`) -- The process index on the current machine.
- **mixed_precision** (`str`) -- The configured mixed precision mode.
- **num_processes** (`int`) -- The total number of processes used for training.
- **optimizer_step_was_skipped** (`bool`) -- Whether or not the optimizer update was skipped (because of
  gradient overflow in mixed precision), in which
case the learning rate should not be changed.
- **process_index** (`int`) -- The overall index of the current process among all processes.
- **state** ([AcceleratorState](/docs/accelerate/v1.11.0/en/package_reference/state#accelerate.state.AcceleratorState)) -- The distributed setup state.
- **sync_gradients** (`bool`) -- Whether the gradients are currently being synced across all processes.
- **use_distributed** (`bool`) -- Whether the current configuration is for distributed training.



<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>accumulate</name><anchor>accelerate.Accelerator.accumulate</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1253</source><parameters>[{"name": "*models", "val": ""}]</parameters><paramsdesc>- ***models** (list of `torch.nn.Module`) --
  PyTorch Modules that were prepared with `Accelerator.prepare`. Models passed to `accumulate()` will
  skip gradient syncing during backward pass in distributed training</paramsdesc><paramgroups>0</paramgroups></docstring>

A context manager that will lightly wrap around and perform gradient accumulation automatically



<ExampleCodeBlock anchor="accelerate.Accelerator.accumulate.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(gradient_accumulation_steps=1)
>>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

>>> for input, output in dataloader:
...     with accelerator.accumulate(model):
...         outputs = model(input)
...         loss = loss_func(outputs)
...         loss.backward()
...         optimizer.step()
...         scheduler.step()
...         optimizer.zero_grad()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>autocast</name><anchor>accelerate.Accelerator.autocast</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L4051</source><parameters>[{"name": "autocast_handler", "val": ": AutocastKwargs = None"}]</parameters></docstring>

Will apply automatic mixed-precision inside the block inside this context manager, if it is enabled. Nothing
different will happen otherwise.

A different `autocast_handler` can be passed in to override the one set in the `Accelerator` object. This is
useful in blocks under `autocast` where you want to revert to fp32.

<ExampleCodeBlock anchor="accelerate.Accelerator.autocast.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(mixed_precision="fp16")
>>> with accelerator.autocast():
...     train()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>backward</name><anchor>accelerate.Accelerator.backward</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2708</source><parameters>[{"name": "loss", "val": ""}, {"name": "**kwargs", "val": ""}]</parameters></docstring>

Scales the gradients in accordance to the `GradientAccumulationPlugin` and calls the correct `backward()` based
on the configuration.

Should be used in lieu of `loss.backward()`.

<ExampleCodeBlock anchor="accelerate.Accelerator.backward.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(gradient_accumulation_steps=2)
>>> outputs = model(inputs)
>>> loss = loss_fn(outputs, labels)
>>> accelerator.backward(loss)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>check_trigger</name><anchor>accelerate.Accelerator.check_trigger</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2768</source><parameters>[]</parameters></docstring>

Checks if the internal trigger tensor has been set to 1 in any of the processes. If so, will return `True` and
reset the trigger tensor to 0.

Note:
Does not require `wait_for_everyone()`

<ExampleCodeBlock anchor="accelerate.Accelerator.check_trigger.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume later in the training script
>>> # `should_do_breakpoint` is a custom function to monitor when to break,
>>> # e.g. when the loss is NaN
>>> if should_do_breakpoint(loss):
...     accelerator.set_trigger()
>>> # Assume later in the training script
>>> if accelerator.check_trigger():
...     break
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>clear</name><anchor>accelerate.Accelerator.clear</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3814</source><parameters>[{"name": "*objects", "val": ""}]</parameters></docstring>

Alias for `Accelerate.free_memory`, releases all references to the internal objects stored and call the
garbage collector. You should call this method between two trainings with different models/optimizers.

<ExampleCodeBlock anchor="accelerate.Accelerator.clear.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer, scheduler = ...
>>> model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
>>> model, optimizer, scheduler = accelerator.clear(model, optimizer, scheduler)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>clip_grad_norm_</name><anchor>accelerate.Accelerator.clip_grad_norm_</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2836</source><parameters>[{"name": "parameters", "val": ""}, {"name": "max_norm", "val": ""}, {"name": "norm_type", "val": " = 2"}]</parameters><rettype>`torch.Tensor`</rettype><retdesc>Total norm of the parameter gradients (viewed as a single vector).</retdesc></docstring>

Should be used in place of `torch.nn.utils.clip_grad_norm_`.





<ExampleCodeBlock anchor="accelerate.Accelerator.clip_grad_norm_.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(gradient_accumulation_steps=2)
>>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

>>> for input, target in dataloader:
...     optimizer.zero_grad()
...     output = model(input)
...     loss = loss_func(output, target)
...     accelerator.backward(loss)
...     if accelerator.sync_gradients:
...         accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
...     optimizer.step()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>clip_grad_value_</name><anchor>accelerate.Accelerator.clip_grad_value_</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2899</source><parameters>[{"name": "parameters", "val": ""}, {"name": "clip_value", "val": ""}]</parameters></docstring>

Should be used in place of `torch.nn.utils.clip_grad_value_`.

<ExampleCodeBlock anchor="accelerate.Accelerator.clip_grad_value_.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(gradient_accumulation_steps=2)
>>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

>>> for input, target in dataloader:
...     optimizer.zero_grad()
...     output = model(input)
...     loss = loss_func(output, target)
...     accelerator.backward(loss)
...     if accelerator.sync_gradients:
...         accelerator.clip_grad_value_(model.parameters(), clip_value)
...     optimizer.step()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>end_training</name><anchor>accelerate.Accelerator.end_training</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3273</source><parameters>[]</parameters></docstring>

Runs any special end training behaviors, such as stopping trackers on the main process only or destoying
process group. Should always be called at the end of your script if using experiment tracking.

<ExampleCodeBlock anchor="accelerate.Accelerator.end_training.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(log_with="tensorboard")
>>> accelerator.init_trackers("my_project")
>>> # Do training
>>> accelerator.end_training()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>free_memory</name><anchor>accelerate.Accelerator.free_memory</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3785</source><parameters>[{"name": "*objects", "val": ""}]</parameters></docstring>

Will release all references to the internal objects stored and call the garbage collector. You should call this
method between two trainings with different models/optimizers. Also will reset `Accelerator.step` to 0.

<ExampleCodeBlock anchor="accelerate.Accelerator.free_memory.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer, scheduler = ...
>>> model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
>>> model, optimizer, scheduler = accelerator.free_memory(model, optimizer, scheduler)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>gather</name><anchor>accelerate.Accelerator.gather</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2926</source><parameters>[{"name": "tensor", "val": ""}]</parameters><paramsdesc>- **tensor** (`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`) --
  The tensors to gather across all processes.</paramsdesc><paramgroups>0</paramgroups><rettype>`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`</rettype><retdesc>The gathered tensor(s). Note that the
first dimension of the result is *num_processes* multiplied by the first dimension of the input tensors.</retdesc></docstring>

Gather the values in *tensor* across all processes and concatenate them on the first dimension. Useful to
regroup the predictions from all processes when doing evaluation.

Note:
This gather happens in all processes.







<ExampleCodeBlock anchor="accelerate.Accelerator.gather.example">

Example:

```python
>>> # Assuming four processes
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> process_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
>>> gathered_tensor = accelerator.gather(process_tensor)
>>> gathered_tensor
tensor([0, 1, 2, 3])
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>gather_for_metrics</name><anchor>accelerate.Accelerator.gather_for_metrics</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2958</source><parameters>[{"name": "input_data", "val": ""}, {"name": "use_gather_object", "val": " = False"}]</parameters><paramsdesc>- **input** (`torch.Tensor`, `object`, a nested tuple/list/dictionary of `torch.Tensor`, or a nested tuple/list/dictionary of `object`) --
  The tensors or objects for calculating metrics across all processes
- **use_gather_object(`bool`)** --
  Whether to forcibly use gather_object instead of gather (which is already done if all objects passed do
  not contain tensors). This flag can be useful for gathering tensors with different sizes that we don't
  want to pad and concatenate along the first dimension. Using it with GPU tensors is not well supported
  and inefficient as it incurs GPU -> CPU transfer since tensors would be pickled.</paramsdesc><paramgroups>0</paramgroups></docstring>

Gathers `input_data` and potentially drops duplicates in the last batch if on a distributed system. Should be
used for gathering the inputs and targets for metric calculation.



<ExampleCodeBlock anchor="accelerate.Accelerator.gather_for_metrics.example">

Example:

```python
>>> # Assuming two processes, with a batch size of 5 on a dataset with 9 samples
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> dataloader = torch.utils.data.DataLoader(range(9), batch_size=5)
>>> dataloader = accelerator.prepare(dataloader)
>>> batch = next(iter(dataloader))
>>> gathered_items = accelerator.gather_for_metrics(batch)
>>> len(gathered_items)
9
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>get_state_dict</name><anchor>accelerate.Accelerator.get_state_dict</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3881</source><parameters>[{"name": "model", "val": ""}, {"name": "unwrap", "val": " = True"}]</parameters><paramsdesc>- **model** (`torch.nn.Module`) --
  A PyTorch model sent through [Accelerator.prepare()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.prepare)
- **unwrap** (`bool`, *optional*, defaults to `True`) --
  Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict</paramsdesc><paramgroups>0</paramgroups><rettype>`dict`</rettype><retdesc>The state dictionary of the model potentially without full precision.</retdesc></docstring>

Returns the state dictionary of a model sent through [Accelerator.prepare()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) potentially without full
precision.







<ExampleCodeBlock anchor="accelerate.Accelerator.get_state_dict.example">

Example:

```python
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> net = torch.nn.Linear(2, 2)
>>> net = accelerator.prepare(net)
>>> state_dict = accelerator.get_state_dict(net)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>get_tracker</name><anchor>accelerate.Accelerator.get_tracker</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3211</source><parameters>[{"name": "name", "val": ": str"}, {"name": "unwrap", "val": ": bool = False"}]</parameters><paramsdesc>- **name** (`str`) --
  The name of a tracker, corresponding to the `.name` property.
- **unwrap** (`bool`) --
  Whether to return the internal tracking mechanism or to return the wrapped tracker instead
  (recommended).</paramsdesc><paramgroups>0</paramgroups><rettype>`GeneralTracker`</rettype><retdesc>The tracker corresponding to `name` if it exists.</retdesc></docstring>

Returns a `tracker` from `self.trackers` based on `name` on the main process only.







<ExampleCodeBlock anchor="accelerate.Accelerator.get_tracker.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(log_with="tensorboard")
>>> accelerator.init_trackers("my_project")
>>> tensorboard_tracker = accelerator.get_tracker("tensorboard")
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>join_uneven_inputs</name><anchor>accelerate.Accelerator.join_uneven_inputs</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1298</source><parameters>[{"name": "joinables", "val": ""}, {"name": "even_batches", "val": " = None"}]</parameters><paramsdesc>- **joinables** (`list[torch.distributed.algorithms.Joinable]`) --
  A list of models or optimizers that subclass `torch.distributed.algorithms.Joinable`. Most commonly, a
  PyTorch Module that was prepared with `Accelerator.prepare` for DistributedDataParallel training.
- **even_batches** (`bool`, *optional*) --
  If set, this will override the value of `even_batches` set in the `Accelerator`. If it is not provided,
  the default `Accelerator` value wil be used.</paramsdesc><paramgroups>0</paramgroups></docstring>

A context manager that facilitates distributed training or evaluation on uneven inputs, which acts as a wrapper
around `torch.distributed.algorithms.join`. This is useful when the total batch size does not evenly divide the
length of the dataset.



<Tip warning={true}>

`join_uneven_inputs` is only supported for Distributed Data Parallel training on multiple GPUs. For any other
configuration, this method will have no effect.

</Tip>

<Tip warning={true}>

Overriding `even_batches` will not affect iterable-style data loaders.

</Tip>

<ExampleCodeBlock anchor="accelerate.Accelerator.join_uneven_inputs.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(even_batches=True)
>>> ddp_model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

>>> with accelerator.join_uneven_inputs([ddp_model], even_batches=False):
...     for input, output in dataloader:
...         outputs = model(input)
...         loss = loss_func(outputs)
...         loss.backward()
...         optimizer.step()
...         optimizer.zero_grad()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>load_state</name><anchor>accelerate.Accelerator.load_state</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3633</source><parameters>[{"name": "input_dir", "val": ": str | None = None"}, {"name": "load_kwargs", "val": ": dict | None = None"}, {"name": "**load_model_func_kwargs", "val": ""}]</parameters><paramsdesc>- **input_dir** (`str` or `os.PathLike`) --
  The name of the folder all relevant weights and states were saved in. Can be `None` if
  `automatic_checkpoint_naming` is used, and will pick up from the latest checkpoint.
- **load_kwargs** (`dict`, *optional*) --
  Additional keyword arguments for the underlying `load` function, such as optional arguments for
  state_dict and optimizer on.
- **load_model_func_kwargs** (`dict`, *optional*) --
  Additional keyword arguments for loading model which can be passed to the underlying load function,
  such as optional arguments for DeepSpeed's `load_checkpoint` function or a `map_location` to load the
  model and optimizer on.</paramsdesc><paramgroups>0</paramgroups></docstring>

Loads the current states of the model, optimizer, scaler, RNG generators, and registered objects.

<Tip>

Should only be used in conjunction with [Accelerator.save_state()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.save_state). If a file is not registered for
checkpointing, it will not be loaded if stored in the directory.

</Tip>



<ExampleCodeBlock anchor="accelerate.Accelerator.load_state.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer, lr_scheduler = ...
>>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
>>> accelerator.load_state("my_checkpoint")
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>local_main_process_first</name><anchor>accelerate.Accelerator.local_main_process_first</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1108</source><parameters>[]</parameters></docstring>

Lets the local main process go inside a with block.

The other processes will enter the with block after the main process exits.

<ExampleCodeBlock anchor="accelerate.Accelerator.local_main_process_first.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> with accelerator.local_main_process_first():
...     # This will be printed first by local process 0 then in a seemingly
...     # random order by the other processes.
...     print(f"This will be printed by process {accelerator.local_process_index}")
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>lomo_backward</name><anchor>accelerate.Accelerator.lomo_backward</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L4194</source><parameters>[{"name": "loss", "val": ": torch.Tensor"}, {"name": "learning_rate", "val": ": float"}]</parameters></docstring>

Runs backward pass on LOMO optimizers.


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>main_process_first</name><anchor>accelerate.Accelerator.main_process_first</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1086</source><parameters>[]</parameters></docstring>

Lets the main process go first inside a with block.

The other processes will enter the with block after the main process exits.

<ExampleCodeBlock anchor="accelerate.Accelerator.main_process_first.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> with accelerator.main_process_first():
...     # This will be printed first by process 0 then in a seemingly
...     # random order by the other processes.
...     print(f"This will be printed by process {accelerator.process_index}")
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>maybe_context_parallel</name><anchor>accelerate.Accelerator.maybe_context_parallel</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3988</source><parameters>[{"name": "buffers", "val": ": list[torch.Tensor] | None = None"}, {"name": "buffer_seq_dims", "val": ": list[int] | None = None"}, {"name": "no_restore_buffers", "val": ": set[torch.Tensor] | None = None"}]</parameters><paramsdesc>- **buffers** (`list[torch.Tensor]`, `optional`) --
  Buffers, which are going to be sharded along the sequence dimension. Common examples are inputs, labels
  or positional embedding buffers. This context manager will modify these buffers in-place, and after
  exiting the context, the buffers will be restored to their original state. To avoid unnecessary
  restores, you can use `no_restore_buffers` to specify which buffers don't need to be restored.
- **buffer_seq_dims** (`list[int]`, `optional`) --
  Sequence dimensions of `buffers`.
- **no_restore_buffers** (`set[torch.Tensor]`, `optional`) --
  This set must be a subset of `buffers`. Specifies which buffers from `buffers` argument won't be
  restored after the context exits. These buffers will be then kept in sharded state.</paramsdesc><paramgroups>0</paramgroups></docstring>

A context manager that enables context parallel training.



<Tip warning={true}>

`context_parallel` is currently only supported together with FSDP2, and requires `parallelism_config.cp_size` >
1. If either of these conditions are not met, this context manager will have no effect, though to enable fewer
code changes it will not raise an Exception.

</Tip>

<Tip warning={true}>

This context manager has to be recreated with each training step, as shown in the example below.

</Tip>

<ExampleCodeBlock anchor="accelerate.Accelerator.maybe_context_parallel.example">

Example:

```python
>>> for batch in dataloader:
...     with accelerator.maybe_context_parallel(
...         buffers=[batch["input_ids"], batch["attention_mask"]],
...         buffer_seq_dims=[1, 1],
...         no_restore_buffers={batch["input_ids"]},
...     ):
...         outputs = model(batch)
...         ...
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>no_sync</name><anchor>accelerate.Accelerator.no_sync</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1130</source><parameters>[{"name": "model", "val": ""}]</parameters><paramsdesc>- **model** (`torch.nn.Module`) --
  PyTorch Module that was prepared with `Accelerator.prepare`</paramsdesc><paramgroups>0</paramgroups></docstring>

A context manager to disable gradient synchronizations across DDP processes by calling
`torch.nn.parallel.DistributedDataParallel.no_sync`.

If `model` is not in DDP, this context manager does nothing



<ExampleCodeBlock anchor="accelerate.Accelerator.no_sync.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)
>>> input_a = next(iter(dataloader))
>>> input_b = next(iter(dataloader))

>>> with accelerator.no_sync():
...     outputs = model(input_a)
...     loss = loss_func(outputs)
...     accelerator.backward(loss)
...     # No synchronization across processes, only accumulate gradients
>>> outputs = model(input_b)
>>> accelerator.backward(loss)
>>> # Synchronization across all processes
>>> optimizer.step()
>>> optimizer.zero_grad()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>on_last_process</name><anchor>accelerate.Accelerator.on_last_process</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L954</source><parameters>[{"name": "function", "val": ": Callable[..., Any]"}]</parameters><paramsdesc>- **function** (`Callable`) -- The function to decorate.</paramsdesc><paramgroups>0</paramgroups></docstring>

A decorator that will run the decorated function on the last process only. Can also be called using the
`PartialState` class.



<ExampleCodeBlock anchor="accelerate.Accelerator.on_last_process.example">

Example:
```python
# Assume we have 4 processes.
from accelerate import Accelerator

accelerator = Accelerator()


@accelerator.on_last_process
def print_something():
    print(f"Printed on process {accelerator.process_index}")


print_something()
"Printed on process 3"
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>on_local_main_process</name><anchor>accelerate.Accelerator.on_local_main_process</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L912</source><parameters>[{"name": "function", "val": ": Callable[..., Any] | None = None"}]</parameters><paramsdesc>- **function** (`Callable`) -- The function to decorate.</paramsdesc><paramgroups>0</paramgroups></docstring>

A decorator that will run the decorated function on the local main process only. Can also be called using the
`PartialState` class.



<ExampleCodeBlock anchor="accelerate.Accelerator.on_local_main_process.example">

Example:
```python
# Assume we have 2 servers with 4 processes each.
from accelerate import Accelerator

accelerator = Accelerator()


@accelerator.on_local_main_process
def print_something():
    print("This will be printed by process 0 only on each server.")


print_something()
# On server 1:
"This will be printed by process 0 only"
# On server 2:
"This will be printed by process 0 only"
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>on_local_process</name><anchor>accelerate.Accelerator.on_local_process</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1038</source><parameters>[{"name": "function", "val": ": Callable[..., Any] | None = None"}, {"name": "local_process_index", "val": ": int | None = None"}]</parameters><paramsdesc>- **function** (`Callable`, *optional*) --
  The function to decorate.
- **local_process_index** (`int`, *optional*) --
  The index of the local process on which to run the function.</paramsdesc><paramgroups>0</paramgroups></docstring>

A decorator that will run the decorated function on a given local process index only. Can also be called using
the `PartialState` class.



<ExampleCodeBlock anchor="accelerate.Accelerator.on_local_process.example">

Example:
```python
# Assume we have 2 servers with 4 processes each.
from accelerate import Accelerator

accelerator = Accelerator()


@accelerator.on_local_process(local_process_index=2)
def print_something():
    print(f"Printed on process {accelerator.local_process_index}")


print_something()
# On server 1:
"Printed on process 2"
# On server 2:
"Printed on process 2"
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>on_main_process</name><anchor>accelerate.Accelerator.on_main_process</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L873</source><parameters>[{"name": "function", "val": ": Callable[..., Any] | None = None"}]</parameters><paramsdesc>- **function** (`Callable`) -- The function to decorate.</paramsdesc><paramgroups>0</paramgroups></docstring>

A decorator that will run the decorated function on the main process only. Can also be called using the
`PartialState` class.



<ExampleCodeBlock anchor="accelerate.Accelerator.on_main_process.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()


>>> @accelerator.on_main_process
... def print_something():
...     print("This will be printed by process 0 only.")


>>> print_something()
"This will be printed by process 0 only"
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>on_process</name><anchor>accelerate.Accelerator.on_process</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L993</source><parameters>[{"name": "function", "val": ": Callable[..., Any] | None = None"}, {"name": "process_index", "val": ": int | None = None"}]</parameters><paramsdesc>- **function** (`Callable`, `optional`) --
  The function to decorate.
- **process_index** (`int`, `optional`) --
  The index of the process on which to run the function.</paramsdesc><paramgroups>0</paramgroups></docstring>

A decorator that will run the decorated function on a given process index only. Can also be called using the
`PartialState` class.



<ExampleCodeBlock anchor="accelerate.Accelerator.on_process.example">

Example:
```python
# Assume we have 4 processes.
from accelerate import Accelerator

accelerator = Accelerator()


@accelerator.on_process(process_index=2)
def print_something():
    print(f"Printed on process {accelerator.process_index}")


print_something()
"Printed on process 2"
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>pad_across_processes</name><anchor>accelerate.Accelerator.pad_across_processes</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3067</source><parameters>[{"name": "tensor", "val": ""}, {"name": "dim", "val": " = 0"}, {"name": "pad_index", "val": " = 0"}, {"name": "pad_first", "val": " = False"}]</parameters><paramsdesc>- **tensor** (nested list/tuple/dictionary of `torch.Tensor`) --
  The data to gather.
- **dim** (`int`, *optional*, defaults to 0) --
  The dimension on which to pad.
- **pad_index** (`int`, *optional*, defaults to 0) --
  The value with which to pad.
- **pad_first** (`bool`, *optional*, defaults to `False`) --
  Whether to pad at the beginning or the end.</paramsdesc><paramgroups>0</paramgroups><rettype>`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`</rettype><retdesc>The padded tensor(s).</retdesc></docstring>

Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
they can safely be gathered.







<ExampleCodeBlock anchor="accelerate.Accelerator.pad_across_processes.example">

Example:

```python
>>> # Assuming two processes, with the first processes having a tensor of size 1 and the second of size 2
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> process_tensor = torch.arange(accelerator.process_index + 1).to(accelerator.device)
>>> padded_tensor = accelerator.pad_across_processes(process_tensor)
>>> padded_tensor.shape
torch.Size([2])
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>prepare</name><anchor>accelerate.Accelerator.prepare</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1413</source><parameters>[{"name": "*args", "val": ""}, {"name": "device_placement", "val": " = None"}]</parameters><paramsdesc>- ***args** (list of objects) --
  Any of the following type of objects:

  - `torch.utils.data.DataLoader`: PyTorch Dataloader
  - `torch.nn.Module`: PyTorch Module
  - `torch.optim.Optimizer`: PyTorch Optimizer
  - `torch.optim.lr_scheduler.LRScheduler`: PyTorch LR Scheduler

- **device_placement** (`list[bool]`, *optional*) --
  Used to customize whether automatic device placement should be performed for each object passed. Needs
  to be a list of the same length as `args`. Not compatible with DeepSpeed or FSDP.</paramsdesc><paramgroups>0</paramgroups></docstring>

Prepare all objects passed in `args` for distributed training and mixed precision, then return them in the same
order.



<Tip>

You don't need to prepare a model if you only use it for inference without any kind of mixed precision

</Tip>

<ExampleCodeBlock anchor="accelerate.Accelerator.prepare.example">

Examples:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume a model, optimizer, data_loader and scheduler are defined
>>> model, optimizer, data_loader, scheduler = accelerator.prepare(model, optimizer, data_loader, scheduler)
```

</ExampleCodeBlock>

<ExampleCodeBlock anchor="accelerate.Accelerator.prepare.example-2">

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume a model, optimizer, data_loader and scheduler are defined
>>> device_placement = [True, True, False, False]
>>> # Will place the first two items passed in automatically to the right device but not the last two.
>>> model, optimizer, data_loader, scheduler = accelerator.prepare(
...     model, optimizer, data_loader, scheduler, device_placement=device_placement
... )
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>prepare_data_loader</name><anchor>accelerate.Accelerator.prepare_data_loader</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2564</source><parameters>[{"name": "data_loader", "val": ": torch.utils.data.DataLoader"}, {"name": "device_placement", "val": " = None"}, {"name": "slice_fn_for_dispatch", "val": " = None"}]</parameters><paramsdesc>- **data_loader** (`torch.utils.data.DataLoader`) --
  A vanilla PyTorch DataLoader to prepare
- **device_placement** (`bool`, *optional*) --
  Whether or not to place the batches on the proper device in the prepared dataloader. Will default to
  `self.device_placement`.
- **slice_fn_for_dispatch** (`Callable`, *optional*`) --
  If passed, this function will be used to slice tensors across `num_processes`. Will default to
  [slice_tensors()](/docs/accelerate/v1.11.0/en/package_reference/utilities#accelerate.utils.slice_tensors). This argument is used only when `dispatch_batches` is set to `True` and will
  be ignored otherwise.</paramsdesc><paramgroups>0</paramgroups></docstring>

Prepares a PyTorch DataLoader for training in any distributed setup. It is recommended to use
[Accelerator.prepare()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) instead.



<ExampleCodeBlock anchor="accelerate.Accelerator.prepare_data_loader.example">

Example:

```python
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> data_loader = torch.utils.data.DataLoader(...)
>>> data_loader = accelerator.prepare_data_loader(data_loader, device_placement=True)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>prepare_model</name><anchor>accelerate.Accelerator.prepare_model</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1707</source><parameters>[{"name": "model", "val": ": torch.nn.Module"}, {"name": "device_placement", "val": ": bool | None = None"}, {"name": "evaluation_mode", "val": ": bool = False"}]</parameters><paramsdesc>- **model** (`torch.nn.Module`) --
  A PyTorch model to prepare. You don't need to prepare a model if it is used only for inference without
  any kind of mixed precision
- **device_placement** (`bool`, *optional*) --
  Whether or not to place the model on the proper device. Will default to `self.device_placement`.
- **evaluation_mode** (`bool`, *optional*, defaults to `False`) --
  Whether or not to set the model for evaluation only, by just applying mixed precision and
  `torch.compile` (if configured in the `Accelerator` object).</paramsdesc><paramgroups>0</paramgroups></docstring>

Prepares a PyTorch model for training in any distributed setup. It is recommended to use
[Accelerator.prepare()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) instead.



<ExampleCodeBlock anchor="accelerate.Accelerator.prepare_model.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume a model is defined
>>> model = accelerator.prepare_model(model)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>prepare_optimizer</name><anchor>accelerate.Accelerator.prepare_optimizer</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2623</source><parameters>[{"name": "optimizer", "val": ": torch.optim.Optimizer"}, {"name": "device_placement", "val": " = None"}]</parameters><paramsdesc>- **optimizer** (`torch.optim.Optimizer`) --
  A vanilla PyTorch optimizer to prepare
- **device_placement** (`bool`, *optional*) --
  Whether or not to place the optimizer on the proper device. Will default to `self.device_placement`.</paramsdesc><paramgroups>0</paramgroups></docstring>

Prepares a PyTorch Optimizer for training in any distributed setup. It is recommended to use
[Accelerator.prepare()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) instead.



<ExampleCodeBlock anchor="accelerate.Accelerator.prepare_optimizer.example">

Example:

```python
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> optimizer = torch.optim.Adam(...)
>>> optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>prepare_scheduler</name><anchor>accelerate.Accelerator.prepare_scheduler</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2667</source><parameters>[{"name": "scheduler", "val": ": LRScheduler"}]</parameters><paramsdesc>- **scheduler** (`torch.optim.lr_scheduler.LRScheduler`) --
  A vanilla PyTorch scheduler to prepare</paramsdesc><paramgroups>0</paramgroups></docstring>

Prepares a PyTorch Scheduler for training in any distributed setup. It is recommended to use
[Accelerator.prepare()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) instead.



<ExampleCodeBlock anchor="accelerate.Accelerator.prepare_scheduler.example">

Example:

```python
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> optimizer = torch.optim.Adam(...)
>>> scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
>>> scheduler = accelerator.prepare_scheduler(scheduler)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>print</name><anchor>accelerate.Accelerator.print</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1381</source><parameters>[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]</parameters></docstring>

Drop in replacement of `print()` to only print once per server.

<ExampleCodeBlock anchor="accelerate.Accelerator.print.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> accelerator.print("Hello world!")
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>profile</name><anchor>accelerate.Accelerator.profile</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L4076</source><parameters>[{"name": "profile_handler", "val": ": ProfileKwargs | None = None"}]</parameters><paramsdesc>- **profile_handler** (`ProfileKwargs`, *optional*) --
  The profile handler to use for this context manager. If not passed, will use the one set in the
  `Accelerator` object.</paramsdesc><paramgroups>0</paramgroups></docstring>

Will profile the code inside the context manager. The profile will be saved to a Chrome Trace file if
`profile_handler.output_trace_dir` is set.

A different `profile_handler` can be passed in to override the one set in the `Accelerator` object.



<ExampleCodeBlock anchor="accelerate.Accelerator.profile.example">

Example:

```python
# Profile with default settings
from accelerate import Accelerator
from accelerate.utils import ProfileKwargs

accelerator = Accelerator()
with accelerator.profile() as prof:
    train()
accelerator.print(prof.key_averages().table())


# Profile with the custom handler
def custom_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


kwargs = ProfileKwargs(schedule_option=dict(wait=1, warmup=1, active=1), on_trace_ready=custom_handler)
accelerator = Accelerator(kwarg_handler=[kwargs])
with accelerator.profile() as prof:
    for _ in range(10):
        train_iteration()
        prof.step()


# Profile and export to Chrome Trace
kwargs = ProfileKwargs(output_trace_dir="output_trace")
accelerator = Accelerator(kwarg_handler=[kwargs])
with accelerator.profile():
    train()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>reduce</name><anchor>accelerate.Accelerator.reduce</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3031</source><parameters>[{"name": "tensor", "val": ""}, {"name": "reduction", "val": " = 'sum'"}, {"name": "scale", "val": " = 1.0"}]</parameters><paramsdesc>- **tensor** (`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`) --
  The tensors to reduce across all processes.
- **reduction** (`str`, *optional*, defaults to "sum") --
  A reduction type, can be one of 'sum', 'mean', or 'none'. If 'none', will not perform any operation.
- **scale** (`float`, *optional*, defaults to 1.0) --
  A default scaling value to be applied after the reduce, only valid on XLA.</paramsdesc><paramgroups>0</paramgroups><rettype>`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`</rettype><retdesc>The reduced tensor(s).</retdesc></docstring>

Reduce the values in *tensor* across all processes based on *reduction*.

Note:
All processes get the reduced value.







<ExampleCodeBlock anchor="accelerate.Accelerator.reduce.example">

Example:

```python
>>> # Assuming two processes
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> process_tensor = torch.arange(accelerator.num_processes) + 1 + (2 * accelerator.process_index)
>>> process_tensor = process_tensor.to(accelerator.device)
>>> reduced_tensor = accelerator.reduce(process_tensor, reduction="sum")
>>> reduced_tensor
tensor([4, 6])
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>register_for_checkpointing</name><anchor>accelerate.Accelerator.register_for_checkpointing</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3952</source><parameters>[{"name": "*objects", "val": ""}]</parameters></docstring>

Makes note of `objects` and will save or load them in during `save_state` or `load_state`.

These should be utilized when the state is being loaded or saved in the same script. It is not designed to be
used in different scripts.

<Tip>

Every `object` must have a `load_state_dict` and `state_dict` function to be stored.

</Tip>

<ExampleCodeBlock anchor="accelerate.Accelerator.register_for_checkpointing.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume `CustomObject` has a `state_dict` and `load_state_dict` function.
>>> obj = CustomObject()
>>> accelerator.register_for_checkpointing(obj)
>>> accelerator.save_state("checkpoint.pt")
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>register_load_state_pre_hook</name><anchor>accelerate.Accelerator.register_load_state_pre_hook</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3602</source><parameters>[{"name": "hook", "val": ": Callable[..., None]"}]</parameters><paramsdesc>- **hook** (`Callable`) --
  A function to be called in [Accelerator.load_state()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.load_state) before `load_checkpoint`.</paramsdesc><paramgroups>0</paramgroups><rettype>`torch.utils.hooks.RemovableHandle`</rettype><retdesc>a handle that can be used to remove the added hook by calling
`handle.remove()`</retdesc></docstring>

Registers a pre hook to be run before `load_checkpoint` is called in [Accelerator.load_state()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.load_state).



The hook should have the following signature:

`hook(models: list[torch.nn.Module], input_dir: str) -> None`

The `models` argument are the models as saved in the accelerator state under `accelerator._models`, and the
`input_dir` argument is the `input_dir` argument passed to [Accelerator.load_state()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.load_state).

<Tip>

Should only be used in conjunction with [Accelerator.register_save_state_pre_hook()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.register_save_state_pre_hook). Can be useful to load
configurations in addition to model weights. Can also be used to overwrite model loading with a customized
method. In this case, make sure to remove already loaded models from the models list.

</Tip>






</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>register_save_state_pre_hook</name><anchor>accelerate.Accelerator.register_save_state_pre_hook</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3435</source><parameters>[{"name": "hook", "val": ": Callable[..., None]"}]</parameters><paramsdesc>- **hook** (`Callable`) --
  A function to be called in [Accelerator.save_state()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.save_state) before `save_checkpoint`.</paramsdesc><paramgroups>0</paramgroups><rettype>`torch.utils.hooks.RemovableHandle`</rettype><retdesc>a handle that can be used to remove the added hook by calling
`handle.remove()`</retdesc></docstring>

Registers a pre hook to be run before `save_checkpoint` is called in [Accelerator.save_state()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.save_state).



The hook should have the following signature:

`hook(models: list[torch.nn.Module], weights: list[dict[str, torch.Tensor]], input_dir: str) -> None`

The `models` argument are the models as saved in the accelerator state under `accelerator._models`, `weights`
argument are the state dicts of the `models`, and the `input_dir` argument is the `input_dir` argument passed
to [Accelerator.load_state()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.load_state).

<Tip>

Should only be used in conjunction with [Accelerator.register_load_state_pre_hook()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.register_load_state_pre_hook). Can be useful to save
configurations in addition to model weights. Can also be used to overwrite model saving with a customized
method. In this case, make sure to remove already loaded weights from the weights list.

</Tip>






</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>save</name><anchor>accelerate.Accelerator.save</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3294</source><parameters>[{"name": "obj", "val": ""}, {"name": "f", "val": ""}, {"name": "safe_serialization", "val": " = False"}]</parameters><paramsdesc>- **obj** (`object`) -- The object to save.
- **f** (`str` or `os.PathLike`) -- Where to save the content of `obj`.
- **safe_serialization** (`bool`, *optional*, defaults to `False`) -- Whether to save `obj` using `safetensors`</paramsdesc><paramgroups>0</paramgroups></docstring>

Save the object passed to disk once per machine. Use in place of `torch.save`.



Note:
If `save_on_each_node` was passed in as a `ProjectConfiguration`, will save the object once per node,
rather than only once on the main node.

<ExampleCodeBlock anchor="accelerate.Accelerator.save.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> arr = [0, 1, 2, 3]
>>> accelerator.save(arr, "array.pkl")
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>save_model</name><anchor>accelerate.Accelerator.save_model</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3324</source><parameters>[{"name": "model", "val": ": torch.nn.Module"}, {"name": "save_directory", "val": ": Union[str, os.PathLike]"}, {"name": "max_shard_size", "val": ": Union[int, str] = '10GB'"}, {"name": "safe_serialization", "val": ": bool = True"}]</parameters><paramsdesc>- **model** -- (`torch.nn.Module`):
  Model to be saved. The model can be wrapped or unwrapped.
- **save_directory** (`str` or `os.PathLike`) --
  Directory to which to save. Will be created if it doesn't exist.
- **max_shard_size** (`int` or `str`, *optional*, defaults to `"10GB"`) --
  The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
  lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

  <Tip warning={true}>

  If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
  which will be bigger than `max_shard_size`.

  </Tip>

- **safe_serialization** (`bool`, *optional*, defaults to `True`) --
  Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).</paramsdesc><paramgroups>0</paramgroups></docstring>

Save a model so that it can be re-loaded using load_checkpoint_in_model



<ExampleCodeBlock anchor="accelerate.Accelerator.save_model.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model = ...
>>> accelerator.save_model(model, save_directory)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>save_state</name><anchor>accelerate.Accelerator.save_state</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3467</source><parameters>[{"name": "output_dir", "val": ": str | None = None"}, {"name": "safe_serialization", "val": ": bool = True"}, {"name": "**save_model_func_kwargs", "val": ""}]</parameters><paramsdesc>- **output_dir** (`str` or `os.PathLike`) --
  The name of the folder to save all relevant weights and states.
- **safe_serialization** (`bool`, *optional*, defaults to `True`) --
  Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
- **save_model_func_kwargs** (`dict`, *optional*) --
  Additional keyword arguments for saving model which can be passed to the underlying save function, such
  as optional arguments for DeepSpeed's `save_checkpoint` function.</paramsdesc><paramgroups>0</paramgroups></docstring>

Saves the current states of the model, optimizer, scaler, RNG generators, and registered objects to a folder.

If a `ProjectConfiguration` was passed to the `Accelerator` object with `automatic_checkpoint_naming` enabled
then checkpoints will be saved to `self.project_dir/checkpoints`. If the number of current saves is greater
than `total_limit` then the oldest save is deleted. Each checkpoint is saved in separate folders named
`checkpoint_<iteration>`.

Otherwise they are just saved to `output_dir`.

<Tip>

Should only be used when wanting to save a checkpoint during training and restoring the state in the same
environment.

</Tip>



<ExampleCodeBlock anchor="accelerate.Accelerator.save_state.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer, lr_scheduler = ...
>>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
>>> accelerator.save_state(output_dir="my_checkpoint")
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>set_trigger</name><anchor>accelerate.Accelerator.set_trigger</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2742</source><parameters>[]</parameters></docstring>

Sets the internal trigger tensor to 1 on the current process. A latter check should follow using this which
will check across all processes.

Note:
Does not require `wait_for_everyone()`

<ExampleCodeBlock anchor="accelerate.Accelerator.set_trigger.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume later in the training script
>>> # `should_do_breakpoint` is a custom function to monitor when to break,
>>> # e.g. when the loss is NaN
>>> if should_do_breakpoint(loss):
...     accelerator.set_trigger()
>>> # Assume later in the training script
>>> if accelerator.check_breakpoint():
...     break
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>skip_first_batches</name><anchor>accelerate.Accelerator.skip_first_batches</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L4147</source><parameters>[{"name": "dataloader", "val": ""}, {"name": "num_batches", "val": ": int = 0"}]</parameters><paramsdesc>- **dataloader** (`torch.utils.data.DataLoader`) -- The data loader in which to skip batches.
- **num_batches** (`int`, *optional*, defaults to 0) -- The number of batches to skip</paramsdesc><paramgroups>0</paramgroups></docstring>

Creates a new `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`.



<ExampleCodeBlock anchor="accelerate.Accelerator.skip_first_batches.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)
>>> skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=2)
>>> # for the first epoch only
>>> for input, target in skipped_dataloader:
...     optimizer.zero_grad()
...     output = model(input)
...     loss = loss_func(output, target)
...     accelerator.backward(loss)
...     optimizer.step()

>>> # subsequent epochs
>>> for input, target in dataloader:
...     optimizer.zero_grad()
...     ...
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>split_between_processes</name><anchor>accelerate.Accelerator.split_between_processes</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L831</source><parameters>[{"name": "inputs", "val": ": list | tuple | dict | torch.Tensor"}, {"name": "apply_padding", "val": ": bool = False"}]</parameters><paramsdesc>- **inputs** (`list`, `tuple`, `torch.Tensor`, or `dict` of `list`/`tuple`/`torch.Tensor`) --
  The input to split between processes.
- **apply_padding** (`bool`, `optional`, defaults to `False`) --
  Whether to apply padding by repeating the last element of the input so that all processes have the same
  number of elements. Useful when trying to perform actions such as `Accelerator.gather()` on the outputs
  or passing in less inputs than there are processes. If so, just remember to drop the padded elements
  afterwards.</paramsdesc><paramgroups>0</paramgroups></docstring>

Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
distributed inference, such as with different prompts.

Note that when using a `dict`, all keys need to have the same number of elements.



<ExampleCodeBlock anchor="accelerate.Accelerator.split_between_processes.example">

Example:

```python
# Assume there are two processes
from accelerate import Accelerator

accelerator = Accelerator()
with accelerator.split_between_processes(["A", "B", "C"]) as inputs:
    print(inputs)
# Process 0
["A", "B"]
# Process 1
["C"]

with accelerator.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
    print(inputs)
# Process 0
["A", "B"]
# Process 1
["C", "C"]
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>trigger_sync_in_backward</name><anchor>accelerate.Accelerator.trigger_sync_in_backward</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L1179</source><parameters>[{"name": "model", "val": ""}]</parameters><paramsdesc>- **model** (`torch.nn.Module`) --
  The model for which to trigger the gradient synchronization.</paramsdesc><paramgroups>0</paramgroups></docstring>
Trigger the sync of the gradients in the next backward pass of the model after multiple forward passes under
`Accelerator.no_sync` (only applicable in multi-GPU scenarios).

If the script is not launched in distributed mode, this context manager does nothing.



<ExampleCodeBlock anchor="accelerate.Accelerator.trigger_sync_in_backward.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)

>>> with accelerator.no_sync():
...     loss_a = loss_func(model(input_a))  # first forward pass
...     loss_b = loss_func(model(input_b))  # second forward pass
>>> accelerator.backward(loss_a)  # No synchronization across processes, only accumulate gradients
>>> with accelerator.trigger_sync_in_backward(model):
...     accelerator.backward(loss_b)  # Synchronization across all processes
>>> optimizer.step()
>>> optimizer.zero_grad()
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>unscale_gradients</name><anchor>accelerate.Accelerator.unscale_gradients</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L2801</source><parameters>[{"name": "optimizer", "val": " = None"}]</parameters><paramsdesc>- **optimizer** (`torch.optim.Optimizer` or `list[torch.optim.Optimizer]`, *optional*) --
  The optimizer(s) for which to unscale gradients. If not set, will unscale gradients on all optimizers
  that were passed to [prepare()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.prepare).</paramsdesc><paramgroups>0</paramgroups></docstring>

Unscale the gradients in mixed precision training with AMP. This is a noop in all other settings.

Likely should be called through [Accelerator.clip_grad_norm_()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_) or [Accelerator.clip_grad_value_()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_value_)



<ExampleCodeBlock anchor="accelerate.Accelerator.unscale_gradients.example">

Example:

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer = accelerator.prepare(model, optimizer)
>>> outputs = model(inputs)
>>> loss = loss_fn(outputs, labels)
>>> accelerator.backward(loss)
>>> accelerator.unscale_gradients(optimizer=optimizer)
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>unwrap_model</name><anchor>accelerate.Accelerator.unwrap_model</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3102</source><parameters>[{"name": "model", "val": ""}, {"name": "keep_fp32_wrapper", "val": ": bool = True"}, {"name": "keep_torch_compile", "val": ": bool = True"}]</parameters><paramsdesc>- **model** (`torch.nn.Module`) --
  The model to unwrap.
- **keep_fp32_wrapper** (`bool`, *optional*, defaults to `True`) --
  Whether to not remove the mixed precision hook if it was added.
- **keep_torch_compile** (`bool`, *optional*, defaults to `True`) --
  Whether to not unwrap compiled model if compiled.</paramsdesc><paramgroups>0</paramgroups><rettype>`torch.nn.Module`</rettype><retdesc>The unwrapped model.</retdesc></docstring>

Unwraps the `model` from the additional layer possible added by [prepare()](/docs/accelerate/v1.11.0/en/package_reference/accelerator#accelerate.Accelerator.prepare). Useful before saving
the model.







<ExampleCodeBlock anchor="accelerate.Accelerator.unwrap_model.example">

Example:

```python
>>> # Assuming two GPU processes
>>> from torch.nn.parallel import DistributedDataParallel
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model = accelerator.prepare(MyModel())
>>> print(model.__class__.__name__)
DistributedDataParallel

>>> model = accelerator.unwrap_model(model)
>>> print(model.__class__.__name__)
MyModel
```

</ExampleCodeBlock>


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>verify_device_map</name><anchor>accelerate.Accelerator.verify_device_map</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L4183</source><parameters>[{"name": "model", "val": ": torch.nn.Module"}]</parameters></docstring>

Verifies that `model` has not been prepared with big model inference with a device-map resembling `auto`.


</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>wait_for_everyone</name><anchor>accelerate.Accelerator.wait_for_everyone</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/accelerator.py#L3136</source><parameters>[]</parameters></docstring>

Will stop the execution of the current process until every other process has reached that point (so this does
nothing when the script is only run in one process). Useful to do before saving a model.

<ExampleCodeBlock anchor="accelerate.Accelerator.wait_for_everyone.example">

Example:

```python
>>> # Assuming two GPU processes
>>> import time
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> if accelerator.is_main_process:
...     time.sleep(2)
>>> else:
...     print("I'm waiting for the main process to finish its sleep...")
>>> accelerator.wait_for_everyone()
>>> # Should print on every process at the same time
>>> print("Everyone is here")
```

</ExampleCodeBlock>


</div></div>

## Utilities[[accelerate.utils.gather_object]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>accelerate.utils.gather_object</name><anchor>accelerate.utils.gather_object</anchor><source>https://github.com/huggingface/accelerate/blob/v1.11.0/src/accelerate/utils/operations.py#L445</source><parameters>[{"name": "object", "val": ": typing.Any"}]</parameters><paramsdesc>- **object** (nested list/tuple/dictionary of picklable object) --
  The data to gather.</paramsdesc><paramgroups>0</paramgroups><retdesc>The same data structure as `object` with all the objects sent to every device.</retdesc></docstring>

Recursively gather object in a nested list/tuple/dictionary of objects from all devices.






</div>

<EditOnGithub source="https://github.com/huggingface/accelerate/blob/main/docs/source/package_reference/accelerator.md" />