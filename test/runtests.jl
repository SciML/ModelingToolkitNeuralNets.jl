using ModelingToolkitNeuralNets
using Test
using SafeTestsets

const GROUP = ENV["GROUP"]

@testset verbose = true "ModelingToolkitNeuralNets.jl" begin
    if GROUP != "Core"
        @safetestset "QA" include("qa.jl")
    end
    @safetestset "Basic" include("lotka_volterra.jl")
    @safetestset "MTK model macro compatibility" include("macro.jl")
    @safetestset "Symbolic Neural Network Macro" include("symbolicnn_macro.jl")
    @safetestset "Neural Network Parameter Metadata" include("nn_ps_accessors.jl")
    @safetestset "Reported issues" include("reported_issues.jl")
end
