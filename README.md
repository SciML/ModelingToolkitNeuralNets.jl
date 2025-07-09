# ModelingToolkitNeuralNets

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SciML.github.io/ModelingToolkitNeuralNets.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SciML.github.io/ModelingToolkitNeuralNets.jl/dev/)
[![Build Status](https://github.com/SciML/ModelingToolkitNeuralNets.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/SciML/ModelingToolkitNeuralNets.jl/actions/workflows/Tests.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SciML/ModelingToolkitNeuralNets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/ModelingToolkitNeuralNets.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

ModelingToolkitNeuralNets.jl is a package to create neural network blocks defined similar to MTKStandardLibrary components, to use them for solving Universal Differential Equations. It can be plugged to any part of the equations in an ODESystem using `RealInputArray` and `RealOutputArray` components which gives a lot of flexibility to add the missing physics only to a part of the model.

## Tutorials and Documentation

For information on using the package, [see the stable documentation](https://docs.sciml.ai/ModelingToolkitNeuralNets/stable/). Use the [in-development documentation](https://docs.sciml.ai/ModelingToolkitNeuralNets/dev/) for the version of the documentation, which contains the unreleased features.

## Breaking changes in v2

The `NeuralNetworkBlock` no longer uses `RealInputArray` & `RealOutputArray`,
the ports are now `inputs` and `outputs` and they are normal vector variables.
This simplifies the usage a bit and removes the need for the ModelingToolkitStandardLibrary dependency.

This version also moves to ModelingToolkit@v10.
