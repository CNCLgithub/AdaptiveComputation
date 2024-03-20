export MotionGauss,
    MotionObs,
    motiongauss

struct MotionObs
    p::SVector{2, Float64}
    a::Float64
end

struct MotionGauss <: Gen.Distribution{MotionObs} end

const motiongauss = MotionGauss()

function Gen.random(::MotionGauss,
                    m::SVector{2, Float64},
                    c::Float64,
                    )
                    # u::Float64,
                    # k::Float64)
    p = Gen.random(broadcasted_normal, m, Fill(c, 2))
    # ang = Gen.random(von_mises, u, k)
    MotionObs(p, 0)
end

const loghalf = log(0.5)

function Gen.logpdf(::MotionGauss,
                    x::MotionObs,
                    m::SVector{2, Float64},
                    c::Float64,
                    )
                    # u::Float64,
                    # k::Float64)
    pl = Gen.logpdf(broadcasted_normal, x.p, m, Fill(c, 2))
    # al = Gen.logpdf(von_mises, x.a, u, k)
    # pl + al
end

(::MotionGauss)(args...) = Gen.random(motiongauss, args...)

Gen.has_output_grad(::MotionGauss) = false
Gen.logpdf_grad(::MotionGauss, value::Set, args...) = (nothing,)
