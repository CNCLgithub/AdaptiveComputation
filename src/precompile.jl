using MOT
using PackageCompiler

cpu_target = PackageCompiler.default_app_cpu_target()
create_sysimage(:MOT, sysimage_path = "/project/mot.so", 
                      precompile_execution_file="test/runtests.jl",
                      cpu_target = cpu_target);
