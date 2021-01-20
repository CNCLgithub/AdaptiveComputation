atreplinit() do repl
    try
        @eval using Revise
        @async Revise.wait_steal_repl_backend()
    catch e
        @warn "Error initializing Revise" exception=(e, catch_backtrace())
    end
end
