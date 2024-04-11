using Test
using ModelingToolkitNeuralNets
using Aqua
using JET

@testset verbose=true "Code quality (Aqua.jl)" begin
    Aqua.find_persistent_tasks_deps(ModelingToolkitNeuralNets)
    Aqua.test_ambiguities(ModelingToolkitNeuralNets, recursive = false)
    Aqua.test_deps_compat(ModelingToolkitNeuralNets)
    Aqua.test_piracies(ModelingToolkitNeuralNets)
    Aqua.test_project_extras(ModelingToolkitNeuralNets)
    Aqua.test_stale_deps(ModelingToolkitNeuralNets, ignore = Symbol[])
    Aqua.test_unbound_args(ModelingToolkitNeuralNets)
    Aqua.test_undefined_exports(ModelingToolkitNeuralNets)
end

@testset "Code linting (JET.jl)" begin
    JET.test_package(ModelingToolkitNeuralNets; target_defined_modules = true)
end
