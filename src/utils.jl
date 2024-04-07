function multi_layer_feed_forward(input_length, output_length; width::Int = 5,
        depth::Int = 1, activation = softplus, disable_optimizations = false)
    Lux.Chain(Lux.Dense(input_length, width, activation),
        [Lux.Dense(width, width, activation) for _ in 1:(depth)]...,
        Lux.Dense(width, output_length); disable_optimizations)
end
