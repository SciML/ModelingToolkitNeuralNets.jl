module ModelingToolkitNeuralNets

using ModelingToolkit: @parameters, @named, ODESystem, t_nounits, @connector, @variables,
                       Equation
using ModelingToolkitStandardLibrary.Blocks: RealInput, RealOutput
using Symbolics: Symbolics, @register_array_symbolic, @wrapped
using LuxCore: stateless_apply
using Lux: Lux
using Random: Xoshiro
using ComponentArrays: ComponentArray

export NeuralNetworkBlock, multi_layer_feed_forward

include("utils.jl")

@connector function RealInput2(; name, nin = 1, u_start = zeros(nin))
    @variables u(t_nounits)[1:nin]=u_start [
        input = true,
        description = "Inner variable in RealInput $name"
    ]
    u = collect(u)
    ODESystem(Equation[], t_nounits, [u...], []; name = name)
end

@connector function RealOutput2(; name, nout = 1, u_start = zeros(nout))
    @variables u(t_nounits)[1:nout]=u_start [
        output = true,
        description = "Inner variable in RealOutput $name"
    ]
    u = collect(u)
    ODESystem(Equation[], t_nounits, [u...], []; name = name)
end

"""
    NeuralNetworkBlock(n_input = 1, n_output = 1;
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64)

Create an `ODESystem` with a neural network inside.
"""
function NeuralNetworkBlock(n_input = 1,
        n_output = 1;
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64)
    ca = ComponentArray{eltype}(init_params)

    @parameters p[1:length(ca)] = Vector(ca)
    @parameters T::typeof(typeof(p))=typeof(p) [tunable = false]

    @named input = RealInput2(nin = n_input)
    @named output = RealOutput2(nout = n_output)

    out = stateless_apply(chain, input.u, lazyconvert(typeof(ca), p))

    eqs = [output.u ~ out]

    @named ude_comp = ODESystem(
        eqs, t_nounits, [], [p, T], systems = [input, output])
    return ude_comp
end

function lazyconvert(T, x::Symbolics.Arr)
    Symbolics.array_term(convert, T, x, size = size(x))
end

end
