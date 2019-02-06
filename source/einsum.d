module einsum;

import std.stdio;
import mir.ndslice;
import mir.primitives : DimensionCount;
import std.typecons : tuple;
import std.format : format;
static import stri;

size_t[dchar] indexLength(dstring indices, S)(S x) if (isSlice!S) {
    static assert(indices.length == DimensionCount!S);
    size_t[dchar] ret;
    static foreach (dim, i; indices) {
        if (i in ret) {
            assert(ret[i] == x.length!dim, "index/length mismatch");
        }
        else {
            ret[i] = x.length!dim;
        }
    }
    return ret;
}

unittest {
    auto m = iota(1, 2, 2);
    // assert(m.indexLength!"ijj" == ['i': 1UL, 'j': 2UL]);
}

auto dim2index(dstring dim2dchar)(size_t[dchar] dchar2index) {
    static if (dim2dchar.length == 0) {
        return tuple();
    }
    else {
        return tuple(dchar2index[dim2dchar[0]],
                     dim2index!(dim2dchar[1..$])(dchar2index).expand);
    }
}

unittest {
    assert(dim2index!"iij"(['i': 1, 'j':2]) == tuple(1, 1, 2));
    assert(dim2index!"i"(['i': 1, 'j':2]) == tuple(1));
}

auto toTuple(T, size_t N)(const auto ref T[N] a) {
    import std.typecons : tuple;
    import std.format : format;
    mixin({
            auto ret = "return tuple(";
            foreach (i; 0 .. N) {
                ret ~=  "a[%d],".format(i);
            }
            return ret[0 .. $-1] ~ ");";
        }());
}

@safe nothrow pure
unittest {
    static assert([1, 2, 3].toTuple == tuple(1, 2, 3));
    static assert("abc".toTuple == tuple('a', 'b', 'c'));
}

auto shapeRanges(dstring dim2dchar, S)(S x) if (isSlice!S) {
    import std.array : array;
    import std.algorithm : sort, uniq, cartesianProduct, map;
    static immutable dchars = dim2dchar.sort.uniq.array;
    size_t[dchars.length] lens;
    auto s = dim2index!dchars(x._lengths);
    return cartesianProduct(x._lengths.expand); // .map!(t => tuple(t.expand));
}

unittest {
    auto m = iota(1, 2, 3);
    // writeln(m.shapeRanges!"ijk");
}

@safe struct Expr {
    dstring[] inputs;
    dstring output;

    void validate() {
        import std.algorithm : canFind;

        foreach (o; output) {
            bool found = false;
            foreach (i; inputs) {
                if (i.canFind(o)) {
                    found = true;
                    break;
                }
            }
            assert(found, format!"symbol %s is not found in input symbols %s"(o, inputs));
        }
    }
}

@safe Expr tokenize(dstring s) {
    import std.algorithm : map, canFind;
    import std.array;
    import std.string;
    assert(!s.empty, "empty expr is invalid in einsum.");
    auto sp = s.split("->");
    assert(sp.length <= 2, "multiple -> is not supported in einsum");

    dstring output;
    if (sp.length == 2) {
        output = sp[1].strip;
        assert(!output.canFind(","), "multiple output is not supported in einsum");
    }

    auto inputs = sp[0].split(",").map!strip.array;
    assert(inputs.length > 0, "at least 1 argument for expr is required in einsum.");
    auto ret = Expr(inputs, output);
    ret.validate();
    return ret;
}

unittest {
    assert(tokenize("ij") == Expr(["ij"]));
    assert(tokenize("ij,ji") == Expr(["ij", "ji"]));
    assert(tokenize("ij,jk->ki") == Expr(["ij", "jk"], "ki"));
}

/**
TODO:
- use swapped/transposed to sort indices
- use opBinary to multiply tensors
- use mir.math.sum to sum elements
- use alongDim/byDim to return slice (unless scalar)
 */
@safe auto einsum(string expr, S...)(S xs) {
    import std.array : empty, join, array;
    import std.algorithm : uniq, sort;
    import std.conv : to;

    static immutable tok = tokenize(expr.to!dstring);
    // reduce to scalar
    static if (tok.output.empty) {
        alias x = xs[0];
        static foreach (i; 0 .. S.length) {
            static assert(isSlice!(S[i]));
            static assert(tok.inputs[i].length == DimensionCount!(S[i]));
        }

        static immutable inSymbols = sort(tok.inputs.join.to!(dchar[])).uniq.to!dstring;
        // TODO replace lengths with tuple
        size_t[dchar] lengths;
        static foreach (narg, input; tok.inputs) {
            static foreach (nsym, c; input) {{
                auto len = xs[narg].length!nsym;
                if (c in lengths) {
                    assert(lengths[c] == len);
                }
                else {
                    lengths[c] = len;
                }
            }}
        }
        // auto lengths = x.indexLength!inDim2Dchar;

        // TODO replace idx with tuple
        size_t[dchar] idx;
        DeepElementType!(S[0]) ret = 0;
        mixin({
                string s;
                // open nested foreach
                static foreach (c; inSymbols) {
                    s ~= format!`
                    foreach (_nested_%s; 0 .. lengths['%s']) {
                        idx['%s'] = _nested_%s;`(c, c, c, c);
                }
                // most inner statement
                s ~= `
                        typeof(ret) tmp = 1;
                        static foreach (_inner_i; 0 .. S.length) {
                            tmp *= xs[_inner_i][dim2index!(tok.inputs[_inner_i])(idx).expand];
                        }
                        ret += tmp;
                `;
                // close nested foreach
                static foreach (c; inSymbols) {
                    s ~= "}";
                }
                return s;
            }());
        return ret;
    } else {
        static assert(false, "not supported expr: " ~ expr);
    }
}

/// TODO support everything in this example
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html

/// trace and scalar reduction
@safe pure nothrow unittest
{
    auto m = iota(5, 5);
    // writeln(m);
    assert(m.einsum!"ii" == 60);
    assert(m.einsum!"ij->" == 300);
    assert(m.einsum!"ji->" == 300);
    assert(einsum!"ii,jj->"(m, m) == 3600);
    assert(einsum!"ii,jk->"(m, m) == 18000);
    assert(einsum!"ii,ij->"(m, m) == 5100);
    assert(einsum!"ii,ji->"(m, m) == 3900);
}
