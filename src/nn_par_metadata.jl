# Defines metadata types.
struct NeuralNetworkParameter end
struct NeuralNetworkParametrisation end
Symbolics.option_to_metadata_type(::Val{:neuralnetwork}) = NeuralNetworkParameter
Symbolics.option_to_metadata_type(::Val{:neuralnetworkps}) = NeuralNetworkParametrisation

# Defines metadata getters.
"""
    ModelingToolkitNeuralNets.isneuralnetwork(p)

Returns `true` if the parameter corresponds to the neural network chain that is saved as a MTK parameter.
"""
isneuralnetwork(p::Union{Symbolics.Num, AbstractVector{Symbolics.Num}}) = isneuralnetwork(Symbolics.value(p))
function isneuralnetwork(p)
    getmetadata(p, NeuralNetworkParameter, false)
end

"""
    ModelingToolkitNeuralNets.isneuralnetworkps(p)

Returns `true` if the parameter corresponds to the a neural network parametrisation.
"""
isneuralnetworkps(p::Union{Symbolics.Num, AbstractVector{Symbolics.Num}}) = isneuralnetworkps(Symbolics.value(p))
function isneuralnetworkps(p)
    getmetadata(p, NeuralNetworkParametrisation, false)
end
