function stress_test_0(T::Int)
    
    optics = []
    
    scale = 1.5

    optics = [optics ; [[[scale*x, 0.0, 0.5],[-scale*x, 0.0, 0.5]] for x=50:-1:1]]
    optics = [optics ; [[[0.0, 0.0, 0.5],[0.0, 0.0, 0.5]] for x=1:1]]
    optics = [optics ; [[[scale*x, 0.0, 0.5],[-scale*x, 0.0, 0.5]] for x=1:50]]

    init_dots = [scale*50.0 0.0 0.5]
    
    return optics, init_dots
end
