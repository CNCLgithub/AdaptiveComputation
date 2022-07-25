using Gen
using Gen_Compose
using GenRFS
using LinearAlgebra

@gen static function particle_init()
    x = @trace(uniform(-1., 1.), :x)
    y = @trace(uniform(-1., 1.), :y)
    result = Float64[x, y]
    return result
end

@gen static function particle_step(particle)
    x, y = particle
    new_x = @trace(uniform(x - 0.5, x + 0.5), :x)
    new_y = @trace(uniform(y - 0.5, y + 0.5), :y)
    new_particle = [x, y]
    return new_particle
end

@gen static function motion_kernel(t::Int64, prev_particles)
    new_particles = @trace(Gen.Map(particle_step)(prev_particles),
                           :particles)
    return new_particles
end

@gen static function gm_particles(k::Int, n::Int)
    init_ps = @trace(Gen.Map(particle_init)(fill((), n)), :init_kernel)
    states = @trace(Gen.Unfold(motion_kernel)(k, init_ps), :kernel)
    result = (init_ps, states)
    return result
end

@load_generated_functions


function main()
    tr, _ = generate(gm_particles, (1, 1))
    addr = :kernel => 1 => :particles => 1 => :x
    display(get_choices(tr))
    res = Gen.regenerate(tr, select(addr))
    res[:]
    # @show res[1]
    # @show res[2]
    # @show res[3]
    cm = choicemap()
    cm[addr] = 0.0
    display(cm)
    res = Gen.update(tr, cm)
    @show res[1]
    @show res[2]
    @show res[3]
    @show res[4]
end

main();
