module ModelingToolkitNeuralNets

using ModelingToolkit: @parameters, @named, ODESystem, t_nounits
using ModelingToolkitStandardLibrary.Blocks: RealInputArray, RealOutputArray
using Symbolics: Symbolics, @register_array_symbolic, @wrapped
using LuxCore: stateless_apply
using Lux: Lux
using Random: Xoshiro
using ComponentArrays: ComponentArray

export NeuralNetworkBlock, multi_layer_feed_forward

include("utils.jl")

"""
    NeuralNetworkBlock(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64,
        name)

Create an `ODESystem` with a neural network inside.
"""
function NeuralNetworkBlock(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64,
        name)
    ca = ComponentArray{eltype}(init_params)

    @parameters p[1:length(ca)] = Vector(ca)
    @parameters T::typeof(typeof(ca))=typeof(ca) [tunable = false]

    @named input = RealInputArray(nin = n_input)
    @named output = RealOutputArray(nout = n_output)

    out = stateless_apply(chain, input.u, lazyconvert(T, p))

    eqs = [output.u ~ out]

    ude_comp = ODESystem(
        eqs, t_nounits, [], [p, T]; systems = [input, output], name)
    return ude_comp
end

# added to avoid a breaking change from moving n_input & n_output in kwargs
# https://github.com/SciML/ModelingToolkitNeuralNets.jl/issues/32
function NeuralNetworkBlock(n_input, n_output = 1; kwargs...)
    NeuralNetworkBlock(; n_input, n_output, kwargs...)
end

function lazyconvert(T, x::Symbolics.Arr)
    Symbolics.array_term(convert, T, x, size = size(x))
end

end
