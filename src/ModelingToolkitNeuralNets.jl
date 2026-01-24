module ModelingToolkitNeuralNets

using ModelingToolkitBase: @parameters, @named, @variables, System, t_nounits
using IntervalSets: var".."
using Symbolics: Symbolics, @register_array_symbolic, @wrapped, unwrap, wrap, shape
using LuxCore: stateless_apply, outputsize
using Lux: Lux
using Random: Xoshiro
using ComponentArrays: ComponentArray

export NeuralNetworkBlock, SymbolicNeuralNetwork, @SymbolicNeuralNetwork, multi_layer_feed_forward, get_network

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
function NeuralNetworkBlock(;
        n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64,
        name
    )
    ca = ComponentArray{eltype}(init_params)

    @parameters p[1:length(ca)] = Vector(ca) [tunable = true]
    @parameters T::typeof(typeof(ca)) = typeof(ca) [tunable = false]
    @parameters lux_model::typeof(chain) = chain [tunable = false]
    @parameters (lux_apply::typeof(stateless_apply))(..)[1:n_output] = stateless_apply [tunable = false]

    @variables inputs(t_nounits)[1:n_input] [input = true]
    @variables outputs(t_nounits)[1:n_output] [output = true]

    expected_outsz = only(outputsize(chain, inputs, rng))
    msg = "The outputsize of the given Lux network ($expected_outsz) does not match `n_output = $n_output`"
    @assert n_output == expected_outsz msg

    eqs = [outputs ~ lux_apply(lux_model, inputs, lazyconvert(T, p))]

    ude_comp = System(
        eqs, t_nounits, [inputs, outputs], [lux_apply, lux_model, p, T]; name
    )
    return ude_comp
end

# added to avoid a breaking change from moving n_input & n_output in kwargs
# https://github.com/SciML/ModelingToolkitNeuralNets.jl/issues/32
function NeuralNetworkBlock(n_input, n_output = 1; kwargs...)
    return NeuralNetworkBlock(; n_input, n_output, kwargs...)
end

function lazyconvert(T, x::Symbolics.Arr)
    return wrap(Symbolics.term(convert, T, unwrap(x); type = Symbolics.getdefaultval(T), shape = shape(x)))
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
function SymbolicNeuralNetwork(;
        n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        nn_name = :NN,
        nn_p_name = :p,
        eltype = Float64
    )
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
    return stateless_apply(get_network(wrapper), input, convert(wrapper.T, nn_p))
end

function (wrapper::StatelessApplyWrapper)(input::Number, nn_p::AbstractVector)
    return wrapper([input], nn_p)
end

function Base.show(io::IO, m::MIME"text/plain", wrapper::StatelessApplyWrapper)
    printstyled(io, "LuxCore.stateless_apply wrapper for:\n", color = :gray)
    return show(io, m, get_network(wrapper))
end

get_network(wrapper::StatelessApplyWrapper) = wrapper.lux_model

"""
    @SymbolicNeuralNetwork

Macro for interfacing with the `SymbolicNeuralNetwork` function. Essentially handles automatic
naming of the symbolic variables. It takes a single input, the Lux chain from which to construct
the symbolic neural network, and returns the corresponding symbolic parameters.

Example:
```
chain = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
)
@SymbolicNeuralNetwork NN, p = chain
```
is equivalent to
```
chain = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
)
NN, p = SymbolicNeuralNetwork(; chain, n_input=1, n_output=1, nn_name =  :NN, nn_p_name = :p)
```
Here, `@SymbolicNeuralNetwork` takes the neural network chain as its input, and:
1) Automatically infer nn_name and nn_p_name from the variable names on the left-hand side of the assignment.
2) Automatically infer n_input and n_output from the chain structure.

Designation of rng. The only other option `@SymbolicNeuralNetwork` currently accepts is a
random number generator. This is simply provided as a second input to `@SymbolicNeuralNetwork`:
```
chain = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 3, Lux.softplus, use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus, use_bias = false)
)
rng = Xoshiro(0)
@SymbolicNeuralNetwork NN, p = chain rng

Notes:
- The first layer of the chain must be one of the following types: `Lux.Dense`, `Lux.Bilinear`,
`Lux.RNNCell`, `Lux.LSTMCell`, `Lux.GRUCell`. For other first layer types, use the `SymbolicNeuralNetwork`
function directly.
```
"""
macro SymbolicNeuralNetwork(expr::Expr)
    return esc(make_symbolic_nn_declaration(expr))
end

# Internal macro for creating the SymbolicNeuralNetwork declaration (as generated by the macro).
function make_symbolic_nn_declaration(expr::Expr)
    # Error checks.
    if !(Meta.isexpr(expr, :(=)) && Meta.isexpr(expr.args[1], :tuple))
        error("@SymbolicNeuralNetwork input ($expr) is poorly erroneously formatted.")
    end
    if length(expr.args[1].args) != 2
        error("@SymbolicNeuralNetwork must have exactly two arguments on the left-hand side. Here, $(length(expr.args[1].args)) arguments were provided.")
    end

    # Extracts individual component symbols.
    nn,p = expr.args[1].args
    chain, rng = if Meta.isexpr(expr.args[2], :tuple)
        if (length(expr.args[2].args) > 2)
            error("@SymbolicNeuralNetwork accepts no more than 2 inputs on the right-hand side.")
        end
        expr.args[2].args
    else
        expr.args[2], nothing
    end

    # Constructs the output expression.
    snn_dec = :(($nn, $p) = SymbolicNeuralNetwork(; chain = $chain, nn_name = $(QuoteNode(nn)),
        nn_p_name = $(QuoteNode(p)), n_input = ModelingToolkitNeuralNets._num_chain_inputs($chain),
        n_output = ModelingToolkitNeuralNets._num_chain_outputs($chain)))
    if !isnothing(rng)
        push!(snn_dec.args[2].args[2].args, Expr(:kw, :rng, rng))
    end
    return snn_dec
end

# Internal functions for determining the number of NN inputs and outputs.
_num_chain_inputs(chain::Lux.Chain) = _num_layer_inputs(chain.layers[1])
_num_chain_inputs(chain) = error("@SymbolicNeuralNetwork has been provided with an input that is not a Lux.Chain.")
_num_layer_inputs(layer::Lux.Dense) = layer.in_dims
_num_layer_inputs(layer::Lux.Bilinear) = layer.in1_dims + layer.in2_dims
_num_layer_inputs(layer::Lux.RNNCell) = layer.in_dims
_num_layer_inputs(layer::Lux.LSTMCell) = layer.in_dims
_num_layer_inputs(layer::Lux.GRUCell) = layer.in_dims
_num_layer_inputs(layer) = error("@SymbolicNeuralNetwork has been provided with a chain which first layer's type ($(typeof(layer))) is not supported for automatic input size detection. Please use the `SymbolicNeuralNetwork` function directly.")
_num_chain_outputs(chain) = chain.layers[end].out_dims

end
