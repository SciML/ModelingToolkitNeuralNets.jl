function multi_layer_feed_forward(input_length, output_length; width::Int = 5,
        depth::Int = 1, activation = softplus)
    Lux.Chain(Lux.Dense(input_length, width, activation),
        [Lux.Dense(width, width, activation) for _ in 1:(depth)]...,
        Lux.Dense(width, output_length); disable_optimizations = true)
end

# Symbolics.@register_array_symbolic print_input(x) begin
#     size = size(x)
#     eltype = eltype(x)
# end

# function print_input(x)
#     @info x
#     x
# end

# function debug_component(n_input, n_output)
#     @named input = RealInput(nin = n_input)
#     @named output = RealOutput(nout = n_output)

#     eqs = [output.u ~ print_input(input.u)]

#     @named dbg_comp = ODESystem(eqs, t_nounits, [], [], systems = [input, output])
# end
