
TARGET_W = 800
TARGET_H = 585 # empirically determined by the aspect ratio
DOT_RADIUS = 15.0

function translate(x, y;
                   min_x = 200, # got these by looking at the
                   max_x = 1500, # empirical distribution
                   min_y = 50,
                   max_y = 1000)

    x -= min_x
    y -= min_y
    
    # making it smaller
    scale_ratio = target_w/(max_x - min_x)
    x *= scale_ratio
    y *= scale_ratio
    
    x -= target_w/2
    y -= target_h/2

    y *= -1  # fliping y

    return [x,y]
end
