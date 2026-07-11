using SciMLTesting, ModelingToolkitNeuralNets
using JET

run_qa(
    ModelingToolkitNeuralNets;
    explicit_imports = true,
    api_docs_kwargs = (; rendered = true),
    ei_kwargs = (;
        # `unwrap`/`shape` are re-exported by Symbolics but owned by SymbolicUtils;
        # `initialparameters` is re-exported by Lux but owned by LuxCore. We use
        # the re-exporting package deliberately (matching the rest of the SciML
        # convention of going through Symbolics / Lux), so ignore the owner
        # mismatch until the names are re-declared public in the re-exporter.
        all_explicit_imports_via_owners = (;
            ignore = (:shape, :unwrap),  # owner SymbolicUtils, from Symbolics
        ),
        all_qualified_accesses_via_owners = (;
            ignore = (
                :initialparameters,  # owner LuxCore, accessed via Lux
                :unwrap,             # owner SymbolicUtils, accessed via Symbolics
            ),
        ),
        # Non-public external names the package genuinely relies on. These are
        # public (via the `public` keyword) on Julia 1.11+, so these ignores are
        # no-ops there and only matter on the LTS (1.10) lane; drop each entry
        # once its owner marks it public on the supported floor.
        all_qualified_accesses_are_public = (;
            ignore = (
                :Arr, :VariableDefaultValue, :option_to_metadata_type, :unwrap,  # Symbolics
                :initialparameters,  # Lux
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :shape, :unwrap, :wrap,       # Symbolics
                :outputsize, :stateless_apply,  # LuxCore
                :t_nounits,                    # ModelingToolkitBase
            ),
        ),
    ),
)
