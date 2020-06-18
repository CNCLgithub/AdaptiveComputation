function stress_test_0(T::Int)
    
    positions = []
    
    scale = 1.5

    positions = [positions ; [[[scale*x, 0.0, 0.5],[-scale*x, 0.0, 0.5]] for x=50:-1:1]]
    positions = [positions ; [[[0.0, 0.0, 0.5],[0.0, 0.0, 0.5]] for x=1:1]]
    positions = [positions ; [[[scale*x, 0.0, 0.5],[-scale*x, 0.0, 0.5]] for x=1:50]]

    init_dots = [scale*50.0 0.0 0.5]
    
    return positions, init_dots
end
