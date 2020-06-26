export visualize

function visualize(xy, full_imgs, params, folder)
    k, n, _, _ = size(xy)

    h, w, ah, aw = params.img_height, params.img_width, params.area_height, params.area_width

    for t=1:k
        img = full_imgs[t]

        for p=1:n
            for i=1:size(xy,3)
                x = xy[t,p,i,1]
                y = xy[t,p,i,2]
                x, y = translate_area_to_img(x, y, h, w, ah, aw)

                draw_circle!(img, [x,y], 5.0, false)
                draw_circle!(img, [x,y], 3.0, true)
                draw_circle!(img, [x,y], 1.0, false)
            end
        end

        mkpath(folder)
        filename = "$(lpad(t, 3, "0")).png"
        save(joinpath(folder, filename), img)
    end
end
