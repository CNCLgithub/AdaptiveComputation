

struct ISRState <: GMState
    walls::SVector{4, Wall}
    objects::Vector{Dot}
end

abstract type GMIncrement{T<:GenerativeModel} end

struct GMIncState{T} <: GMIncrement{T}
    gm::T
    state::GMState{T}
    forces::Matrix{SVector{2, Float64}}
end

struct GMIncremental{T} <: CustomUpdateGF{GMState{T}, GMIncState{T}} end

# step
function Gen.apply_with_state(::GMIncremental{T}, args) where {T}
    # TODO: alternative parsing scheme?
    gm::T = args[1]
    prev = args[2]
    next_state = step(gm, prev)
    inc_state = ISRIncState{T}(gm, next_state)
    state = MyState(arr, s)
    (s, state)
end

function Gen.update_with_state(::GMIncremental{T}, state, args,
                               argdiffs::Tuple{<:Diff, <:VectorDiff}) where {T}
    gm::T = args[1]
    objects = args[2]

    forces = state.forces
    changed = state.changed

    arr = args[1]
    prev_sum = state.sum
    retval = prev_sum
    # compute new forces 
    for i in keys(argdiffs[1].updated)
        changed[i] = true
        for j = something # TODO
            fij = @view forces[i, j]
            force!(fij, objects[i], objects[j])
            forces[j, i] = forces[i, j]
            changed[j] = true
        end
    end

    # update accumulated forces
    for (i, c) = enumerate(changed)
        c || continue
        facc = sum(forces[:, i])
        object = objects[i]
        ku = update_kinematics(gm, object, facc)
        new_objects[i] = sync_update(object, ku)
    end
    state = MyState(arr, retval)
    (state, retval, UnknownChange())
end

Gen.num_args(::MySum) = 1
