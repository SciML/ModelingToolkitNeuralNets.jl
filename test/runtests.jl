using UDEComponents
using Test
using Aqua
using JET

@testset "UDEComponents.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(UDEComponents)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(UDEComponents; target_defined_modules = true)
    end
    # Write your tests here.
end
