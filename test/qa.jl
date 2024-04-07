using Test
using UDEComponents
using Aqua
using JET

@testset verbose=true "Code quality (Aqua.jl)" begin
    Aqua.find_persistent_tasks_deps(UDEComponents)
    Aqua.test_ambiguities(UDEComponents, recursive = false)
    Aqua.test_deps_compat(UDEComponents)
    # TODO: fix type piracy in propagate_ndims and propagate_shape
    Aqua.test_piracies(UDEComponents, broken = true)
    Aqua.test_project_extras(UDEComponents)
    Aqua.test_stale_deps(UDEComponents, ignore = Symbol[])
    Aqua.test_unbound_args(UDEComponents)
    Aqua.test_undefined_exports(UDEComponents)
end

@testset "Code linting (JET.jl)" begin
    JET.test_package(UDEComponents; target_defined_modules = true)
end
