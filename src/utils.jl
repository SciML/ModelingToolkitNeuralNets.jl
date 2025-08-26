"""
    multi_layer_feed_forward(; n_input, n_output, width::Int = 4,
        depth::Int = 1, activation = tanh, use_bias = true, initial_scaling_factor = 1e-8)

Create a Lux.jl `Chain` for use in [`NeuralNetworkBlock`](@ref)s. The weights of the last layer
are multiplied by the `initial_scaling_factor` in order to make the initial contribution
of the network small and thus help with achieving a stable starting position for the training.
"""
function multi_layer_feed_forward(; n_input, n_output, width::Int = 4,
        depth::Int = 1, activation = tanh, use_bias = true, initial_scaling_factor = 1e-8)
    Lux.Chain(
        Lux.Dense(n_input, width, activation; use_bias),
        [Lux.Dense(width, width, activation; use_bias) for _ in 1:(depth)]...,
        Lux.Dense(width, n_output;
            init_weight = (
                rng, a...) -> initial_scaling_factor *
                              Lux.kaiming_uniform(rng, a...), use_bias)
    )
end

function multi_layer_feed_forward(n_input, n_output; kwargs...)
    multi_layer_feed_forward(; n_input, n_output, kwargs...)
end
