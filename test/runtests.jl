using UDEComponents
using Test
using SafeTestsets

@testset verbose=true "UDEComponents.jl" begin
    @safetestset "QA" include("qa.jl")
    @safetestset "Basic" include("lotka_volterra.jl")
end
