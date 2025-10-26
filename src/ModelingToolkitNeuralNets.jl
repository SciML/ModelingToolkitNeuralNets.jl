module ModelingToolkitNeuralNets

using ModelingToolkit: @parameters, @named, @variables, System, t_nounits
using IntervalSets: var".."
using Symbolics: Symbolics, @register_array_symbolic, @wrapped
using LuxCore: stateless_apply, outputsize
using Lux: Lux
using Random: Xoshiro
using ComponentArrays: ComponentArray

export NeuralNetworkBlock, SymbolicNeuralNetwork, multi_layer_feed_forward, get_network

include("utils.jl")

"""
    NeuralNetworkBlock(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64,
        name)

Create a component neural network as a `System`.
"""
function NeuralNetworkBlock(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64,
        name)
    ca = ComponentArray{eltype}(init_params)

    @parameters p[1:length(ca)]=Vector(ca) [tunable = true]
    @parameters T::typeof(typeof(ca))=typeof(ca) [tunable = false]
    @parameters lux_model::typeof(chain)=chain [tunable = false]

    @variables inputs(t_nounits)[1:n_input] [input = true]
    @variables outputs(t_nounits)[1:n_output] [output = true]

    expected_outsz = only(outputsize(chain, inputs, rng))
    msg = "The outputsize of the given Lux network ($expected_outsz) does not match `n_output = $n_output`"
    @assert n_output==expected_outsz msg

    eqs = [outputs ~ stateless_apply(lux_model, inputs, lazyconvert(T, p))]

    ude_comp = System(
        eqs, t_nounits, [inputs, outputs], [lux_model, p, T]; name)
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

"""
    SymbolicNeuralNetwork(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        nn_name =  :NN,
        nn_p_name = :p,
        eltype = Float64)

Create symbolic parameter for a neural network and one for its parameters.
Example:

```
chain = multi_layer_feed_forward(2, 2)
NN, p = SymbolicNeuralNetwork(; chain, n_input=2, n_output=2, rng = StableRNG(42))
```

The NN and p are symbolic parameters that can be used later as part of a system.
To change the name of the symbolic variables, use `nn_name` and `nn_p_name`.
To get the predictions of the neural network, use

```
pred ~ NN(input, p)
```

where `pred` and `input` are a symbolic vector variable with the lengths `n_output` and `n_input`.

To use this outside of an equation, you can get the default values for the symbols and make a similar call

```
defaults(sys)[sys.NN](input, nn_p)
```

where `sys` is a system (e.g. `ODESystem`) that contains `NN`, `input` is a vector of `n_input` length and
`nn_p` is a vector representing parameter values for the neural network.

To get the underlying Lux model you can use `get_network(defaults(sys)[sys.NN])` or
"""
function SymbolicNeuralNetwork(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        nn_name = :NN,
        nn_p_name = :p,
        eltype = Float64)
    ca = ComponentArray{eltype}(init_params)
    wrapper = StatelessApplyWrapper(chain, typeof(ca))

    p = @parameters $(nn_p_name)[1:length(ca)] = Vector(ca)
    NN = @parameters ($(nn_name)::typeof(wrapper))(..)[1:n_output] = wrapper

    return only(NN), only(p)
end

struct StatelessApplyWrapper{NN}
    lux_model::NN
    T::DataType
end

function (wrapper::StatelessApplyWrapper)(input::AbstractArray, nn_p::AbstractVector)
    stateless_apply(get_network(wrapper), input, convert(wrapper.T, nn_p))
end

function (wrapper::StatelessApplyWrapper)(input::Number, nn_p::AbstractVector)
    wrapper([input], nn_p)
end

function Base.show(io::IO, m::MIME"text/plain", wrapper::StatelessApplyWrapper)
    printstyled(io, "LuxCore.stateless_apply wrapper for:\n", color = :gray)
    show(io, m, get_network(wrapper))
end

get_network(wrapper::StatelessApplyWrapper) = wrapper.lux_model

end
