using CSV
using DataFrames
using UnicodePlots

# 1. Compute centroid per frame and per chain
# 2. Aggregate across chains
# 3. Explore time window
#
# Raw data format:
# frame	tracker	importance	cycles	pred_x	pred_y	scene	chain

function importance_centroid(tracker, importance, pred_x, pred_y)
    mu_x = sum(pred_x .* importance)
    mu_y = sum(pred_y .* importance)
    (ic_x = mu_x, ic_y = mu_y)
end

function label_probe(scene, frame, pdf)
    spdf = pdf[pdf.scene .== scene, :]
    probe = 0
    for r in eachrow(spdf)
        probe = r.tracker
        r.frame >= frame && break
    end
    probe
end

function distance_to_centroid(cx, cy, x, y)
    @. sqrt((cx - x)^2 + (cy - y)^2)
end

df = DataFrame(CSV.File("/spaths/test/att.csv"))
pdf = DataFrame(CSV.File("/spaths/test/probes.csv"))

scf = groupby(df, Cols(:scene, :chain, :frame))
# aggregate x and y across trackers
# scene    chain   frame   icx  icy
ic = combine(scf,
     [:tracker, :importance, :pred_x, :pred_y] => importance_centroid =>
         AsTable)

s1 = filter(x -> x.scene == 1, ic)
c1 = filter(:chain => (x -> x == 1), s1)
# scene   chain   frame   icx  icy    probe
label_probe_f = ByRow((x,y) -> label_probe(x,y,pdf))
# transform(c1, [:scene, :frame] => label_probe_f => :probe)
transform!(ic, [:scene, :frame] => label_probe_f => :probe)
filter!(:probe => x -> x > 0, ic)

distances = select(df, Not(Cols(:importance, :cycles)))
display(distances)
# distances = distances[distances.tracker .== distances.probe]

# scene   chain   frame   tracker   pred_x   pred_y   icx  icy   probe
distances = innerjoin(ic, distances, on = [:scene, :frame, :chain])
display(distances)
distances = distances[distances.tracker .== distances.probe, All()]

# scene   chain   frame   pred_x   pred_y   icx  icy
select!(distances, Not(Cols(:tracker, :probe)))

# scene   chain   frame   pred_x   pred_y   icx  icy   d_ic
transform!(distances,
           [:ic_x, :ic_y, :pred_x, :pred_y] => distance_to_centroid => :d_ic)

# scene   chain   frame   ic_x  ic_y   d_ic
select!(distances, Not(Cols(:pred_x, :pred_y)))
display(distances)


# a sanity check, looking at all chains on one scene
s1 = filter(x -> x.scene == 1, distances)
c1 = filter(:chain => (x -> x == 1), s1)
plt = scatterplot(c1.frame, c1.d_ic, name = "chain 1",
                  height = 100, width = 100)
for i = 2:10
    ci = filter(:chain => (x -> x == i), s1)
    scatterplot!(plt, ci.frame, ci.d_ic, name = "chain $i")
end
display(plt)
