using PackageCompiler

create_sysimage(:MOT,
                sysimage_path="/project/mot.so",
                cpu_target = "generic",
                precompile_execution_file="/project/test/case_studies/correlated_sensitivity.jl");
# create_sysimage([:Gen, :Luxor, :Gadfly, :LightGraphs, :MetaGraphs],
#                 sysimage_path="/project/deps.so",
#                 precompile_execution_file="/project/scripts/deps_compile.jl");
