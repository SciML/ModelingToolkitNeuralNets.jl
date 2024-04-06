using UDEComponents
using Documenter

DocMeta.setdocmeta!(UDEComponents, :DocTestSetup, :(using UDEComponents); recursive=true)

makedocs(;
    modules=[UDEComponents],
    authors="Sebastian Micluța-Câmpeanu <sebastian.mc95@proton.me> and contributors",
    sitename="UDEComponents.jl",
    format=Documenter.HTML(;
        canonical="https://SebastianM-C.github.io/UDEComponents.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SebastianM-C/UDEComponents.jl",
    devbranch="main",
)
