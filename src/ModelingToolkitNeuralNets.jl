module ModelingToolkitNeuralNets

using ModelingToolkitBase: @parameters, @variables, System, t_nounits, getdefault,
    getmetadata
using IntervalSets: var".."
using Symbolics: Symbolics, unwrap, wrap, shape
using LuxCore: stateless_apply, outputsize
using Lux: Lux
using Random: Xoshiro
using ComponentArrays: ComponentArray

export NeuralNetworkBlock, SymbolicNeuralNetwork, @SymbolicNeuralNetwork, multi_layer_feed_forward, get_network

include("utils.jl")

# Functionality for accessing various neural network-related parameter properties.
include("nn_par_accessors.jl")

"""
    NeuralNetworkBlock(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        eltype = Float64,
        name)
    NeuralNetworkBlock(n_input, n_output = 1; kwargs...)

Create a ModelingToolkit component system that evaluates a Lux neural network.

# Arguments

  - `n_input`: Number of scalar inputs accepted by the network.
  - `n_output`: Number of scalar outputs produced by the network.

# Keyword Arguments

  - `chain`: Lux model to call from the generated symbolic equations.
  - `rng`: Random number generator used to initialize parameters and query the Lux output size.
  - `init_params`: Initial Lux parameter container. It is flattened into the tunable
    parameter vector `p`.
  - `eltype`: Element type used for the stored `ComponentArray` parameter values.
  - `name`: Required ModelingToolkit component name.

# Returns

A `System` with input variables `inputs`, output variables `outputs`, tunable
network parameters `p`, and non-tunable parameters storing the Lux model and
parameter-container type.

# Examples

```julia
using Lux, ModelingToolkitBase, ModelingToolkitNeuralNets, Random

chain = multi_layer_feed_forward(2, 1; width = 8, depth = 2)
@named nn = NeuralNetworkBlock(2, 1; chain, rng = Xoshiro(0))

length(nn.inputs) == 2
length(nn.outputs) == 1
```
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
    ca_tag = CATypeTag{typeof(ca)}()

    @parameters p[1:length(ca)] = Vector(ca) [tunable = true, neuralnetworkps = true]
    @parameters T::typeof(ca_tag) = ca_tag [tunable = false]
    @parameters lux_model::typeof(chain) = chain [tunable = false]
    @parameters (lux_apply::typeof(stateless_apply))(..)[1:n_output] = stateless_apply [tunable = false, neuralnetwork = true]

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

struct CATypeTag{CAT} end

_ca_type(::CATypeTag{CAT}) where {CAT} = CAT
@inline Base.convert(::CATypeTag{CAT}, x) where {CAT} = convert(CAT, x)

function lazyconvert(T, x::Symbolics.Arr)
    CAT = _ca_type(getdefault(T))
    return wrap(Symbolics.term(convert, T, unwrap(x); type = CAT, shape = shape(x)))
end

"""
    SymbolicNeuralNetwork(; n_input = 1, n_output = 1,
        chain = multi_layer_feed_forward(n_input, n_output),
        rng = Xoshiro(0),
        init_params = Lux.initialparameters(rng, chain),
        nn_name =  :NN,
        nn_p_name = :p,
        eltype = Float64)

Create a callable symbolic neural-network parameter and a symbolic parameter vector.

# Keyword Arguments

  - `n_input`: Number of scalar entries expected in the network input vector.
  - `n_output`: Number of scalar entries returned by the network.
  - `chain`: Lux model represented by the returned callable parameter.
  - `rng`: Random number generator used to initialize Lux parameters.
  - `init_params`: Initial Lux parameter container. It is flattened into the returned
    symbolic parameter vector.
  - `nn_name`: Symbol used as the callable neural-network parameter name.
  - `nn_p_name`: Symbol used as the neural-network parameter-vector name.
  - `eltype`: Element type used for the stored `ComponentArray` parameter values.

# Returns

A tuple `(NN, p)` where `NN(input, p)` is a symbolic callable parameter and `p`
is a symbolic vector containing the flattened neural-network parameters.

# Interface

`NN(input, p)` expects `input` to have length `n_input` and `p` to have the same
length as the flattened `init_params`. The returned symbolic expression has
length `n_output`. Use [`get_network`](@ref) on the default value of `NN` to
recover the underlying Lux model.

# Examples

```julia
using Lux, ModelingToolkitBase, ModelingToolkitNeuralNets, Random

chain = multi_layer_feed_forward(2, 2; width = 4)
NN, p = SymbolicNeuralNetwork(; chain, n_input = 2, n_output = 2, rng = Xoshiro(0))

get_network(ModelingToolkitBase.getdefault(NN)) === chain
```

The returned values can be used in equations as symbolic parameters:

```julia
using ModelingToolkitBase

@variables x(t_nounits)[1:2] y(t_nounits)[1:2]
eqs = [y ~ NN(x, p)]
```
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
    wrapper = StatelessApplyWrapper{typeof(chain), typeof(ca)}(chain)

    p = @parameters $(nn_p_name)[1:length(ca)] = Vector(ca) [tunable = true, neuralnetworkps = true]
    NN = @parameters ($(nn_name)::typeof(wrapper))(..)[1:n_output] = wrapper [tunable = false, neuralnetwork = true]

    return only(NN), only(p)
end

struct StatelessApplyWrapper{NN, CAT}
    lux_model::NN
end

function (wrapper::StatelessApplyWrapper{NN, CAT})(
        input::AbstractArray, nn_p::AbstractVector
    ) where {NN, CAT}
    return stateless_apply(get_network(wrapper), input, convert(CAT, nn_p))
end

function (wrapper::StatelessApplyWrapper)(input::Number, nn_p::AbstractVector)
    return wrapper([input], nn_p)
end

function Base.show(io::IO, m::MIME"text/plain", wrapper::StatelessApplyWrapper)
    printstyled(io, "LuxCore.stateless_apply wrapper for:\n", color = :gray)
    return show(io, m, get_network(wrapper))
end

"""
    get_network(wrapper)

Return the Lux model stored by a symbolic neural-network callable.

# Arguments

  - `wrapper`: A callable wrapper obtained from the default value of a neural-network
    parameter created by [`SymbolicNeuralNetwork`](@ref).

# Returns

The Lux model passed as the `chain` keyword when constructing the symbolic neural network.

# Examples

```julia
using Lux, ModelingToolkitBase, ModelingToolkitNeuralNets

chain = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus; use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus; use_bias = false),
)
NN, p = SymbolicNeuralNetwork(; chain, n_input = 1, n_output = 1)

get_network(ModelingToolkitBase.getdefault(NN)) === chain
```
"""
get_network(wrapper::StatelessApplyWrapper) = wrapper.lux_model

"""
    @SymbolicNeuralNetwork
    @SymbolicNeuralNetwork NN, p = chain
    @SymbolicNeuralNetwork NN, p = chain rng

Construct a symbolic neural network while inferring names and dimensions from the assignment.

# Arguments

  - `NN`: Left-hand-side name for the callable neural-network parameter.
  - `p`: Left-hand-side name for the flattened neural-network parameter vector.
  - `chain`: Lux chain whose first and last layers determine `n_input` and `n_output`.
  - `rng`: Optional random number generator passed to [`SymbolicNeuralNetwork`](@ref).

# Interface

The macro supports Lux chains whose first and last layers are `Lux.Dense`, because
those layers expose the input and output dimensions needed for automatic size
inference. For other layer types, call [`SymbolicNeuralNetwork`](@ref) directly
with explicit `n_input` and `n_output`.

# Examples

```julia
using Lux, ModelingToolkitNeuralNets, Random

chain = Lux.Chain(
    Lux.Dense(1 => 3, Lux.softplus; use_bias = false),
    Lux.Dense(3 => 1, Lux.softplus; use_bias = false),
)
rng = Xoshiro(0)
@SymbolicNeuralNetwork NN, p = chain rng
```
"""
macro SymbolicNeuralNetwork(expr::Expr)
    return esc(make_symbolic_nn_declaration(expr))
end

# Internal macro for creating the SymbolicNeuralNetwork declaration (as generated by the macro).
function make_symbolic_nn_declaration(expr::Expr)
    # Error checks.
    if !(Meta.isexpr(expr, :(=)) && Meta.isexpr(expr.args[1], :tuple))
        error("@SymbolicNeuralNetwork input ($expr) is erroneously formatted.")
    end
    if length(expr.args[1].args) != 2
        error("@SymbolicNeuralNetwork must have exactly two arguments on the left-hand side. Here, $(length(expr.args[1].args)) arguments were provided.")
    end

    # Extracts individual component symbols.
    nn, p = expr.args[1].args
    chain, rng = if Meta.isexpr(expr.args[2], :tuple)
        if (length(expr.args[2].args) > 2)
            error("@SymbolicNeuralNetwork accepts no more than 2 inputs on the right-hand side.")
        end
        expr.args[2].args
    else
        expr.args[2], nothing
    end

    # Constructs the output expression.
    snn_dec = :(
        ($nn, $p) = SymbolicNeuralNetwork(;
            chain = $chain, nn_name = $(QuoteNode(nn)),
            nn_p_name = $(QuoteNode(p)), n_input = ModelingToolkitNeuralNets._num_chain_inputs($chain),
            n_output = ModelingToolkitNeuralNets._num_chain_outputs($chain)
        )
    )
    if !isnothing(rng)
        push!(snn_dec.args[2].args[2].args, Expr(:kw, :rng, rng))
    end
    return snn_dec
end

# Internal functions for determining the number of NN inputs and outputs.
_num_chain_inputs(chain::Lux.Chain) = _num_layer_inputs(chain.layers[1])
_num_chain_inputs(chain) = error("@SymbolicNeuralNetwork has been provided with an input that is not a Lux.Chain.")
_num_layer_inputs(layer::Lux.Dense) = layer.in_dims
_num_layer_inputs(layer) = error("@SymbolicNeuralNetwork has been provided with a chain which first layer's type ($(typeof(layer))) is not supported for automatic input size detection. Please use the `SymbolicNeuralNetwork` function directly.")
_num_chain_outputs(chain::Lux.Chain) = _num_layer_outputs(chain.layers[end])
_num_layer_outputs(layer::Lux.Dense) = layer.out_dims
_num_layer_outputs(layer) = error("@SymbolicNeuralNetwork has been provided with a chain which last layer's type ($(typeof(layer))) is not supported for automatic output size detection. Please use the `SymbolicNeuralNetwork` function directly.")

# Layer types that can potentially be supported in the future.
# _num_layer_inputs(layer::Lux.Bilinear) = layer.in1_dims + layer.in2_dims
# _num_layer_inputs(layer::Lux.RNNCell) = layer.in_dims
# _num_layer_inputs(layer::Lux.LSTMCell) = layer.in_dims
# _num_layer_inputs(layer::Lux.GRUCell) = layer.in_dims

end
