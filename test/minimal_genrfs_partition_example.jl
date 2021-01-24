using GenRFS

rfs_elements = RFSElements{Bool}(undef, 3)
rfs_elements[1] = PoissonElement{Bool}(0.00000001, bernoulli, (0.5,))
rfs_elements[2] = BernoulliElement{Bool}(0.99, bernoulli, (0.5,))
rfs_elements[3] = BernoulliElement{Bool}(0.99, bernoulli, (0.5,))

xs = fill(true, 2)
record = AssociationRecord(10)
Gen.logpdf(rfs, xs, rfs_elements, record)
display(record)

