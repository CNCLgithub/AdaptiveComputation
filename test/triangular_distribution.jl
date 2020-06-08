using Distributions

a = -400
b = 400

dist_x = TriangularDist(a, b, 50)
dist_y = TriangularDist(a, b, -20)
r = 20

count = 0
runs = 10

center_pdf = pdf(dist_x, 50)*pdf(dist_y, -20)
scaling = 0.01/center_pdf
#scaling = sqrt(pi*r^2)

for run=1:runs
    print("run: $run \r")
    for i=-400:400
        for j=-400:400
            p = pdf(dist_x,i)*pdf(dist_y,j)
            p *= scaling
            
            if bernoulli(p)
                global count += 1
            end
        end
    end
end

count /= runs

println("expected pixels: $count")
println("wanted quantity: $(pi*r^2)")
