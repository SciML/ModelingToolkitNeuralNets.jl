"""
    multi_layer_feed_forward(; n_input, n_output, width::Int = 4,
        depth::Int = 1, activation = tanh, use_bias = true, initial_scaling_factor = 1e-8)
    multi_layer_feed_forward(n_input, n_output; kwargs...)

Create a fully connected Lux chain for symbolic neural-network models.

# Arguments

  - `n_input`: Number of scalar inputs to the first dense layer.
  - `n_output`: Number of scalar outputs from the final dense layer.

# Keyword Arguments

  - `width`: Number of hidden units in each hidden dense layer.
  - `depth`: Number of hidden dense layers after the first hidden layer.
  - `activation`: Activation function used by the hidden dense layers.
  - `use_bias`: Whether each dense layer includes a bias vector.
  - `initial_scaling_factor`: Multiplicative factor applied to the final layer's
    initial weights.

# Returns

A `Lux.Chain` compatible with [`NeuralNetworkBlock`](@ref),
[`SymbolicNeuralNetwork`](@ref), and [`@SymbolicNeuralNetwork`](@ref).

# Examples

```julia
using ModelingToolkitNeuralNets

chain = multi_layer_feed_forward(2, 1; width = 8, depth = 2, activation = tanh)
```
"""
function multi_layer_feed_forward(;
        n_input, n_output, width::Int = 4,
        depth::Int = 1, activation = tanh, use_bias = true, initial_scaling_factor = 1.0e-8
    )
    return Lux.Chain(
        Lux.Dense(n_input, width, activation; use_bias),
        [Lux.Dense(width, width, activation; use_bias) for _ in 1:(depth)]...,
        Lux.Dense(
            width, n_output;
            init_weight = (
                rng, a...,
            ) -> initial_scaling_factor *
                Lux.kaiming_uniform(rng, a...), use_bias
        )
    )
end

function multi_layer_feed_forward(n_input, n_output; kwargs...)
    return multi_layer_feed_forward(; n_input, n_output, kwargs...)
end
