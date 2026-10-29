using SciMLTesting, ModelingToolkitNeuralNets

# `EnzymeRules.inactive_type` is the documented Enzyme API, but EnzymeCore does not declare it public.
run_qa(
    ModelingToolkitNeuralNets;
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = (:EnzymeRules,)),
        all_qualified_accesses_are_public = (; ignore = (:inactive_type,)),
    )
)
