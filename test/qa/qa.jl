using SciMLTesting, ModelingToolkitNeuralNets, Test
using JET

run_qa(
    ModelingToolkitNeuralNets;
    explicit_imports = true,
    ei_kwargs = (;
        # Symbolics internals (not marked public) used for symbolic term/metadata
        # manipulation; ignore until Symbolics marks them public.
        all_qualified_accesses_are_public = (;
            ignore = (:CallAndWrap, :SymbolicT, :getdefaultval),  # source: Symbolics
        ),
    ),
)
