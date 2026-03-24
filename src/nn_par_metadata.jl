# Defines metadata types.
struct NeuralNetworkParameter end
struct NeuralNetworkParametrisation end
Symbolics.option_to_metadata_type(::Val{:neuralnetwork}) = NeuralNetworkParameter
Symbolics.option_to_metadata_type(::Val{:neuralnetworkps}) = NeuralNetworkParametrisation

# Defines metadata getters.
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
