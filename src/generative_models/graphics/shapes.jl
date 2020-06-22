export draw_circle!

# drawing a circle in img
function draw_circle!(img, center, radius, value)
    for i=1:size(img,1)
        for j=1:size(img,2)
            if dist(center, [i,j]) < radius
                img[j,i] = value
            end
        end
    end
end
