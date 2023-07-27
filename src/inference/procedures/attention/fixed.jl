export FixedResources

@with_kw struct FixedResources <: AbstractAttentionModel
    # number of unique latents to attend to
    latents::Int = 4
    base_samples::Int = 10
    init_samples = base_samples
    percept_update::Function
    percept_args::Tuple
end

function load(::Type{FixedResources}, path; kwargs...)
    FixedResources(;read_json(path)..., kwargs...)
end

function AdaptiveComputation(att::FixedResources)
    n = att.latents
    base_arrousal = n * att.init_samples
    base_importance = fill(1 / n, n)
    AdaptiveComputation(sensitivities = zeros(n),
                        importance = base_importance,
                        arrousal = base_arrousal)
end

# returns the sensitivity of each latent variable
function hypothesis_testing!(chain::PFChain, att::FixedResources)
    @unpack proc, state, auxillary = chain
    @unpack base_samples, latents = att
    # number of particles
    np = length(state.traces)
    # counter for acceptance ratio
    c = 0
    accepted = 0
    for l = 1:latents # for each latent
        for i = 1:np # for each particle
            # initialize objective of S -> P
            s = state.traces[i]
            for j = 1:base_samples
                # perceptual update:: S -> (S', dS)
                s_prime, ls = att.percept_update(s, l , att.percept_args...)
                c +=1
                if log(rand()) < ls
                    # accepted a proposal and update references
                    accepted += 1
                    s = s_prime
                end
            end
            state.traces[i] = s
        end
    end
    acceptance = accepted / c
    @pack! auxillary = acceptance
    println("acceptance ratio $(acceptance)")
    return nothing
end

function update_importance!(chain::PFChain, att::FixedResources)
    return nothing
end

function update_arrousal!(chain::PFChain, att::FixedResources)
    return nothing
end
