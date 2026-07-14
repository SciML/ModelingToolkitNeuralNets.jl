### Defines Metadata Type ###
struct NeuralNetworkParameter end
struct NeuralNetworkParametrisation end
Symbolics.option_to_metadata_type(::Val{:neuralnetwork}) = NeuralNetworkParameter
Symbolics.option_to_metadata_type(::Val{:neuralnetworkps}) = NeuralNetworkParametrisation

_symbolic_metadata(p, key, default) = _symbolic_metadata_unwrapped(Symbolics.unwrap(p), key, default)
function _symbolic_metadata_unwrapped(p, key, default)
    return applicable(getmetadata, p, key, default) ? getmetadata(p, key, default) : default
end

### Defines Metadata Getters ###
"""
    ModelingToolkitNeuralNets.isneuralnetwork(p)

Return whether `p` is the symbolic callable for a neural network.

# Arguments

  - `p`: Symbolic variable, parameter, symbolic array, or callable symbolic wrapper to inspect.

# Returns

`true` when `p` has neural-network callable metadata and `false` otherwise.

# Examples

```julia
using Lux, ModelingToolkitBase, ModelingToolkitNeuralNets

chain = multi_layer_feed_forward(1, 1)
@SymbolicNeuralNetwork NN, θ = chain

ModelingToolkitNeuralNets.isneuralnetwork(NN)
ModelingToolkitNeuralNets.isneuralnetwork(θ)
```
"""
isneuralnetwork(p) = _symbolic_metadata(p, NeuralNetworkParameter, false)

"""
    ModelingToolkitNeuralNets.hasneuralnetwork(p)

Returns `true` if the parameter has the `neuralnetwork` metadata set (whenever the value is `true` or `false`). This function is primarily intended for internal use within dependent packages.

Example:
```julia
@parameters d
@SymbolicNeuralNetwork NN, θ = chain
ModelingToolkitNeuralNets.hasneuralnetwork(d) # false
ModelingToolkitNeuralNets.hasneuralnetwork(NN) # true
ModelingToolkitNeuralNets.hasneuralnetwork(θ) # false
````
"""
hasneuralnetwork(p::Union{Symbolics.Num, Symbolics.Arr, Symbolics.CallAndWrap}) = hasneuralnetwork(Symbolics.unwrap(p))
function hasneuralnetwork(p::Symbolics.SymbolicT)
    hasmetadata(p, NeuralNetworkParameter)
end

"""
    ModelingToolkitNeuralNets.isneuralnetworkps(p)

Return whether `p` is the symbolic parameter vector for a neural network.

# Arguments

  - `p`: Symbolic variable, parameter, symbolic array, or callable symbolic wrapper to inspect.

# Returns

`true` when `p` has neural-network parameter-vector metadata and `false` otherwise.

# Examples

```julia
using Lux, ModelingToolkitBase, ModelingToolkitNeuralNets

chain = multi_layer_feed_forward(1, 1)
@SymbolicNeuralNetwork NN, θ = chain

ModelingToolkitNeuralNets.isneuralnetworkps(NN)
ModelingToolkitNeuralNets.isneuralnetworkps(θ)
```
"""
isneuralnetworkps(p) = _symbolic_metadata(p, NeuralNetworkParametrisation, false)

"""
    ModelingToolkitNeuralNets.hasneuralnetworkps(p)

Returns `true` if the parameter has the `neuralnetworkps` metadata set (whenever the value is `true` or `false`). This function is primarily intended for internal use within dependent packages.

Example:
```julia
@parameters d
@SymbolicNeuralNetwork NN, θ = chain
ModelingToolkitNeuralNets.hasneuralnetworkps(d) # false
ModelingToolkitNeuralNets.hasneuralnetworkps(NN) # false
ModelingToolkitNeuralNets.hasneuralnetworkps(θ) # true
````
"""
hasneuralnetworkps(p::Union{Symbolics.Num, Symbolics.Arr, Symbolics.CallAndWrap}) = hasneuralnetworkps(Symbolics.unwrap(p))
function hasneuralnetworkps(p::Symbolics.SymbolicT)
    hasmetadata(p, NeuralNetworkParametrisation)
end


### Defines Other Accessors ###

"""
    ModelingToolkitNeuralNets.get_nn_chain(p)

Return the Lux chain associated with a symbolic neural-network callable.

# Arguments

  - `p`: Symbolic callable parameter created by [`SymbolicNeuralNetwork`](@ref) or
    [`@SymbolicNeuralNetwork`](@ref).

# Returns

The Lux chain stored as the default value of `p`.

# Throws

Throws an `ErrorException` when `p` is not a neural-network callable parameter.

# Examples

```julia
using ModelingToolkitNeuralNets

chain = multi_layer_feed_forward(1, 1)
@SymbolicNeuralNetwork NN, θ = chain

ModelingToolkitNeuralNets.get_nn_chain(NN) === chain
```
"""
get_nn_chain(p) = _get_nn_chain_unwrapped(Symbolics.unwrap(p))
function _get_nn_chain_unwrapped(p)
    isneuralnetwork(p) || error("Parameter $p does not have a neural network chain associated with it.")
    return getmetadata(p, Symbolics.VariableDefaultValue).lux_model
end
