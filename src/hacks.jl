lazyconvert(x, y) = convert(x, y)
lazyconvert(x, y::Symbolics.Arr) = Symbolics.array_term(convert, x, y)
Symbolics.propagate_ndims(::typeof(convert), x, y) = ndims(y)
Symbolics.propagate_shape(::typeof(convert), x, y) = Symbolics.shape(y)
