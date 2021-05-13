abstract type GraphicalEnsemble end

"""
    ensemble that is used to generate
    a set of noisy bitmatrices with poisson rate
    and certain pixel probability of turning on
"""
mutable struct PPP <: GraphicalEnsemble
    ppp_rate::Float64
    ppp_pixel_prob::Float64
end


"""
    calculates ppp rate and ppp pixel prob for each receptive field
    based on ensemble rate
"""
function get_rf_ppp_rate_pixel_prob(ensemble::PPP,
                                    rf::AbstractReceptiveField,
                                    graphics::Graphics)

    # this is not completely correct, but perhaps a fine approximation
    rf_proportion_of_img = prod(get_dimensions(rf))/prod(graphics.img_dims)
    rf_ppp_rate = ensemble.ppp_rate * rf_proportion_of_img
    rf_ppp_pixel_prob = ensemble.ppp_pixel_prob * rf_proportion_of_img 
    
    return rf_ppp_rate, rf_ppp_pixel_prob
end
