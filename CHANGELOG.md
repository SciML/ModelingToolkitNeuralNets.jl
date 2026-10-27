# ModelingToolkitNeuralNets v3

## Breaking changes

  - `NeuralNetworkBlock` now stores the Lux model in a single callable parameter,
    `lux_apply`, like `SymbolicNeuralNetwork`. The `lux_model` and `T` parameters were
    removed and the equation is now `outputs ~ lux_apply(inputs, p)`.
    To evaluate the network outside the system, use `nn = sol.ps[sys.nn.lux_apply]`
    and call `nn(input, p)`, or get the Lux model with `get_nn_chain(sys.nn.lux_apply)`.

## Improvements

  - `multi_layer_feed_forward` scales the last layer's initial weights with a callable
    struct (`ScaledInit`, marked inactive for Enzyme) instead of a closure capturing the
    scale. Its chains are now constant for Enzyme, so Enzyme no longer shadows and
    rebuilds them on every gradient evaluation. The initial weights are unchanged.
  - `NeuralNetworkBlock` now rebuilds its parameter `ComponentArray` on the storage of
    `p`, so device arrays stay on device (same as `SymbolicNeuralNetwork` since #162).
  - `get_nn_chain` now works on the `lux_apply` parameter of a `NeuralNetworkBlock`.
  - `NeuralNetworkBlock` no longer fails for models whose output size Lux cannot infer
    with `outputsize`, such as models converted with `ToSimpleChainsAdaptor`; the
    output size check is skipped for them.

# ModelingToolkitNeuralNets v2

## Breaking changes

  - The `NeuralNetworkBlock` no longer uses `RealInputArray` & `RealOutputArray`,
    the ports are now `inputs` and `outputs` and they are normal vector variables.
