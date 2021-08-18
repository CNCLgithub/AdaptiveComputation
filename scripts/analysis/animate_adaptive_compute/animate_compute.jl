using MOT
using FileIO


function plot_compute()
	compute = retrieve_rej_attempts("test.jld2")
	compute .+= 10
	#compute = vcat(zeros(50), compute, zeros(50))


	nt = length(compute)
	for t=1:nt
		print(t, "\r")
		plot_rejuvenation(compute, t=t)
	end
end

plot_compute()
