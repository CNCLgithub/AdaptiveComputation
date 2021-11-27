export LogBernoulliElement

struct LogBernoulliElement{T} <: GenRFS.MonomorphicRFE{T}
    lfr::Float64
    d::Gen.Distribution{T}
    args::Tuple
end

GenRFS.distribution(rfe::LogBernoulliElement) = rfe.d
GenRFS.args(rfe::LogBernoulliElement) = rfe.args

function GenRFS.cardinality(rfe::LogBernoulliElement, n::Int)
    r = log(-expm1(rfe.lfr))
    n > 1 ? -Inf : (n === 1 ? r : rfe.lfr)
end

function GenRFS.sample_cardinality(rfe::LogBernoulliElement)
    Int64(rand() > (expm1(rfe.lfr) + 1.))
end
