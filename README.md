# Some Topological inspired pooling.
For each time step, the module computes

F(X) = ψ(b + ⟨A, CDF( H(X; W(X)) )⟩),

where W(X) = Φ(ctx(X))V are data-adaptive directions derived from permutation-invariant cross-attention ctx(X). (this can be trivially made perm informed)

H(X;W) are heights x_s⊤w_d, and CDF is the z-scored empirical CDF over the set dimension S.
Hence F is permutation-invariant in S, bounded, piecewise-smooth a.e.  It is, compactly, a learned, adaptive distributional summary of a set.


Result of toying with euler characteristics to campture global shape of vector fields, it captures the overall 'shape' of the input via creating input dependent 'views' of the entire input patch without the big gridwise matmuls. 

And pools it to a fixed length embedding.
```
@misc{algomancer2025,
  author = {@algomancer},
  title  = {Some Dumb Shit},
  year   = {2025}
}
```
