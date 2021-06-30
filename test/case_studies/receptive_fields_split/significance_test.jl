function significance_test(as::Vector{Float64}, bs::Vector{Float64};
                           runs = 100000)
    
    bigger = Vector{Bool}(undef, runs)
    for i=1:runs
        println(i)
        idx_a = rand(1:length(as), 100)
        idx_b = rand(1:length(as), 100)
        
        a_mu = mean(as[idx_a])
        b_mu = mean(bs[idx_b])

        bigger[i] = a_mu > b_mu
    end
    
    return sum(bigger)/runs
end

