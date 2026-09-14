using SafeTestsets
using SciMLTesting

run_tests(;
    core = function ()
        @safetestset "Precompile workload" include("precompile_workload.jl")
        @safetestset "Basic" include("lotka_volterra.jl")
        @safetestset "MTK model macro compatibility" include("macro.jl")
        @safetestset "Symbolic Neural Network Macro" include("symbolicnn_macro.jl")
        @safetestset "Neural Network Parameter Metadata" include("nn_ps_accessors.jl")
        return @safetestset "Reported issues" include("reported_issues.jl")
    end,
    groups = Dict(
        "CUDA" => (;
            env = joinpath(@__DIR__, "gpu"),
            body = joinpath(@__DIR__, "gpu", "cuda_tests.jl"),
        ),
    ),
    qa = joinpath(@__DIR__, "qa", "qa.jl"),
)
