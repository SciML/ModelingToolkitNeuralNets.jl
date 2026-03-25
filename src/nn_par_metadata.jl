# Defines metadata types.
struct NeuralNetworkParameter end
struct NeuralNetworkParametrisation end
Symbolics.option_to_metadata_type(::Val{:neuralnetwork}) = NeuralNetworkParameter
Symbolics.option_to_metadata_type(::Val{:neuralnetworkps}) = NeuralNetworkParametrisation

### Defines Metadata Getters ###
"""
    ModelingToolkitNeuralNets.isneuralnetwork(p)

Returns `true` if the parameter corresponds to the neural network chain that is saved as a MTK parameter.

Example:
```julia
@parameters d
@SymbolicNeuralNetwork NN, θ = chain
ModelingToolkitNeuralNets.isneuralnetwork(d) # false
ModelingToolkitNeuralNets.isneuralnetwork(NN) # true
ModelingToolkitNeuralNets.isneuralnetwork(θ) # false
````
"""
isneuralnetwork(p::Union{Symbolics.Num, Symbolics.Arr, Symbolics.CallAndWrap}) = isneuralnetwork(Symbolics.unwrap(p))
function isneuralnetwork(p::Symbolics.SymbolicT)
    getmetadata(p, NeuralNetworkParameter, false)
end

"""
    ModelingToolkitNeuralNets.hasneuralnetwork(p)

Returns `true` if the parameter has the `neuralnetwork` metadata set (whenever the value is `true` or `false`).

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

Returns `true` if the parameter corresponds to the a neural network parametrisation.

Example:
```julia
@parameters d
@SymbolicNeuralNetwork NN, θ = chain
ModelingToolkitNeuralNets.isneuralnetworkps(d) # false
ModelingToolkitNeuralNets.isneuralnetworkps(NN) # false
ModelingToolkitNeuralNets.isneuralnetworkps(θ) # true
````
"""
isneuralnetworkps(p::Union{Symbolics.Num, Symbolics.Arr, Symbolics.CallAndWrap}) = isneuralnetworkps(Symbolics.unwrap(p))
function isneuralnetworkps(p::Symbolics.SymbolicT)
    getmetadata(p, NeuralNetworkParametrisation, false)
end

"""
    ModelingToolkitNeuralNets.hasneuralnetworkps(p)

Returns `true` if the parameter has the `neuralnetworkps` metadata set (whenever the value is `true` or `false`).

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
