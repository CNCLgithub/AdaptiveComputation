using ArrayFire
using Profile
using StatProfilerHTML

size = 400
function test()
    var = 2.0

    for i=1:10
        image = rand(size, size)
        mean = rand(size, size)
        diff = image - mean

        lpdfs = -(diff .* diff)/(2.0 * var) .- 0.5 * log(2.0 * pi * var)
        lpdf = sum(lpdfs)
    end
end

function test_gpu()
    var = 2.0

    for i=1:10
        image = rand(size, size)
        mean = rand(size, size)

        image_gpu = AFArray(image)
        mean_gpu = AFArray(mean)
        image_gpu = image_gpu - mean_gpu
        
        lpdfs_gpu = -(image_gpu .* image_gpu)/(2.0 * var) .- 0.5 * log(2.0 * pi * var)
        lpdf_gpu = sum(lpdfs_gpu)

        lpdf = Float64(lpdf_gpu)
    end
end

setafgcthreshold(2)

@profilehtml begin
    test()
    test_gpu()
end
