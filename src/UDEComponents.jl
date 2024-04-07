module UDEComponents

using ModelingToolkit: @parameters, @named, ODESystem, t_nounits
using ModelingToolkitStandardLibrary.Blocks: RealInput, RealOutput
using Symbolics: Symbolics, @register_array_symbolic, @wrapped
using LuxCore: stateless_apply
using Lux: Lux
using Random: Xoshiro
using ComponentArrays: ComponentArray

export create_ude_component, multi_layer_feed_forward

include("utils.jl")

"""
    create_ude_component(n_input = 1, n_output = 1;
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0))

Create an `ODESystem` with a neural network inside.
"""
function create_ude_component(n_input = 1,
        n_output = 1;
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0))
    lux_p, st = Lux.setup(rng, chain)
    ca = ComponentArray(lux_p)

    @parameters p[1:length(ca)] = Vector(ca)
    @parameters T::typeof(typeof(p))=typeof(p) [tunable = false]

    @named input = RealInput(nin = n_input)
    @named output = RealOutput(nout = n_output)

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
