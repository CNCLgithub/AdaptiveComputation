using MOT
using PackageCompiler
create_sysimage(:MOT, sysimage_path = "/project/mot.so", precompile_execution_file="test/runtests.jl");
