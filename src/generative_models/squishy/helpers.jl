using LinearAlgebra

# converts distance between neighboring vertices to polygon radius
function d_to_r_pol(d, n)
    r = d/sqrt((cos(2*pi/n) - cos(4*pi/n))^2 + (sin(2*pi/n) - sin(4*pi/n))^2)
    println("d $d, n $n, r $r")
    return r
end

function contains(p, hgm)
    xmin, xmax = (-hgm.area_width/2, hgm.area_width/2)
    ymin, ymax = (-hgm.area_height/2, hgm.area_height/2)

    xcheck = p[1] >= xmin && p[1] <= xmax
    ycheck = p[2] >= ymin && p[2] <= ymax
    return xcheck && ycheck
end

function get_wall_normal(p1, p2, hgm)
    wall_vec = p2 .- p1
    x = wall_vec[2]/norm(wall_vec)
    y = -wall_vec[1]/norm(wall_vec)
    n = [x,y]
    return contains(n, hgm) ? n : -n
end


"""
    creates Wall object from tuples p1, p2
"""
function init_wall(p1, p2, hgm)
    Wall([p1...], [p2...], get_wall_normal(p1, p2, hgm))
end

function init_walls(hgm::HGMParams)
    # getting wall points
    wp = @>> Iterators.product((1,-1), (1, -1)) begin
        map(x -> x .* (hgm.area_width/2, hgm.area_height/2))
    end
    
    # getting the walls
    ws = Vector{Wall}(undef, 4)
    ws[1] = init_wall(wp[1], wp[3], hgm) # top r -> bot r
    ws[2] = init_wall(wp[3], wp[4], hgm) # bot r -> bot l
    ws[3] = init_wall(wp[4], wp[2], hgm) # bot l -> top l
    ws[4] = init_wall(wp[2], wp[1], hgm) # top l -> top r
    
    return ws
end
