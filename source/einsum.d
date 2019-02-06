module einsum;

@safe:

import std.stdio;
import mir.ndslice;
import mir.primitives : DimensionCount;
import std.typecons : tuple, isTuple;
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

pure nothrow dim2index(dstring dim2dchar, D2I)(const auto ref D2I dchar2index) {
    static if (dim2dchar.length == 0) {
        return tuple();
    }
    else {
        return tuple(dchar2index.get!(dim2dchar[0]),
                     dim2index!(dim2dchar[1..$])(dchar2index).expand);
    }
}

unittest {
    assert(dim2index!"iij"(tuple!("i", "j")(1, 2)) == tuple(1, 1, 2));
    assert(dim2index!"i"(tuple!("i", "j")(1, 2)) == tuple(1));
}


@safe struct Expr {
    dstring[] inputs;
    dstring output;

    dstring inSymbols() pure const {
        import std.string : join;
        import std.algorithm : sort, uniq;
        import std.conv : to;
        return inputs.join.to!(dchar[]).sort.uniq.to!dstring;
    }

    void validate() pure const {
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

pure Expr tokenize(dstring s) {
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


template NamedArray(T, alias names) {
    import std.typecons : Tuple;
    mixin(
        {
            string ret = "alias NamedArray = Tuple!(";
            foreach (n; names) {
                ret ~= format!"T, \"%s\","(n);
            }
            return ret[0 .. $-1] ~ ");";
        }()
        );
}

unittest {
    NamedArray!(int, "abcあ"d) x;
    assert(x.a == 0);
    x.a = 1;
    x.あ = 2;

    NamedArray!(int, ["a", "あ"]) y;
    y.a = 1;
    y.あ = 2;
}

ref get(alias key, T)(return ref T t) if (isTuple!T) {
    mixin(format!"return t.%s;"(key));
}

@safe unittest {
    auto t = tuple!("a", "b")(1, 2);
    assert(t.get!"a" == 1);
    assert(t.get!'a' == 1);
    t.get!"a" = 3;
    assert(t.a == 3);
    assert(t.get!'b' == 2);
}

auto sym2len(Expr expr, S...)(S xs) {
    static immutable inSymbols = expr.inSymbols;
    NamedArray!(size_t, inSymbols) lengths;
    static foreach (narg, input; expr.inputs) {
        static foreach (nsym, c; input) {{
                auto len = xs[narg].length!nsym;
                if (lengths.get!c != 0) {
                    assert(lengths.get!c == len);
                }
                else {
                    lengths.get!c = len;
                }
            }}
    }
    return lengths;
}


/**
TODO: support output tensor (not only scalar)
- use swapped/transposed to sort indices
- use opBinary to multiply tensors
- use mir.math.sum to sum elements
- use alongDim/byDim to return slice (unless scalar)
 */
auto einsum(string expr, S...)(S xs) {
    import std.range : empty;
    import std.conv : to;

    static immutable tok = tokenize(expr.to!dstring);
    static foreach (i; 0 .. S.length) {
        static assert(isSlice!(S[i]));
        static assert(tok.inputs[i].length == DimensionCount!(S[i]));
    }

    // reduce to scalar
    static if (tok.output.empty) {
        auto lengths = sym2len!tok(xs);
        static immutable inSymbols = tok.inSymbols;
        NamedArray!(size_t, inSymbols) idx;
        DeepElementType!(S[0]) ret = 0;

        mixin({
                string s;
                // open nested foreach
                static foreach (c; inSymbols) {{
                    s ~= format!`
                    foreach (_nested_%s; 0 .. lengths.%s) {
                        idx.%s = _nested_%s;`(c, c, c, c);
                }}
                // most inner statement
                s ~= `  typeof(ret) tmp = 1;
                        static foreach (_inner_i; 0 .. S.length) {
                            tmp *= xs[_inner_i][dim2index!(tok.inputs[_inner_i])(idx).expand];
                        }
                        ret += tmp;`;
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
@nogc @safe pure nothrow unittest
{
    auto m = iota(5, 5);
    assert(m.einsum!"ii" == 60);
    assert(m.einsum!"ああ" == 60);
    assert(m.einsum!"ij->" == 300);
    assert(m.einsum!"ji->" == 300);
    assert(einsum!"ii,jj->"(m, m) == 3600);
    assert(einsum!"ii,jk->"(m, m) == 18000);
    assert(einsum!"ii,ij->"(m, m) == 5100);
    assert(einsum!"ii,ji->"(m, m) == 3900);
    assert(einsum!"ああ,iあ->"(m, m) == 3900);
}
