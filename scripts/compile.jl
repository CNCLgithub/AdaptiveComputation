using PackageCompiler

# create_sysimage(:MOT, sysimage_path="/project/deps.so", precompile_execution_file="/project/test/runtests.jl");
create_sysimage([:Gen, :Luxor, :Gadfly, :LightGraphs, :MetaGraphs],
                sysimage_path="/project/deps.so",
                precompile_execution_file="/project/scripts/deps_compile.jl");
