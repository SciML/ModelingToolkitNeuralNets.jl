using ModelingToolkitNeuralNets
using Test
using SafeTestsets

@testset verbose=true "ModelingToolkitNeuralNets.jl" begin
    @safetestset "QA" include("qa.jl")
    @safetestset "Basic" include("lotka_volterra.jl")
end
