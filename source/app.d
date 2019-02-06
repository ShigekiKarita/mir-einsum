import std.stdio;
import mir.ndslice;
import mir.primitives : DimensionCount;
import std.typecons : tuple;
import std.format : format;

size_t[char] indexLength(string indices, S)(S x) if (isSlice!S) {
    static assert(indices.length == DimensionCount!S);
    size_t[char] ret;
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
    assert(m.indexLength!"ijj" == ['i': 1U, 'j': 2U]);
}

auto dim2index(string dim2char)(size_t[char] char2index) {
    static if (dim2char.length == 0) {
        return tuple();
    }
    else {
        return tuple(char2index[dim2char[0]],
                     dim2index!(dim2char[1..$])(char2index).expand);
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

auto shapeRanges(string dim2char, S)(S x) if (isSlice!S) {
    import std.array : array;
    import std.algorithm : uniq, cartesianProduct, map;
    static immutable chars = dim2char.uniq.array;
    size_t[chars.length] lens;
    auto s = dim2index!chars(x._lengths);
    return cartesianProduct(x._lengths.expand); // .map!(t => tuple(t.expand));
}

unittest {
    auto m = iota(1, 2, 3);
    // writeln(m.shapeRanges!"ijk");
}

struct Expr {
    string[] inputs;
    string output;

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

Expr tokenize(string s) {
    import std.algorithm : map, canFind;
    import std.array;
    import std.string;
    assert(!s.empty, "empty expr is invalid in einsum.");
    auto sp = s.split("->");
    assert(sp.length <= 2, "multiple -> is not supported in einsum");

    string output;
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
auto einsum(string expr, S...)(S xs) if (isSlice!S) {
    import std.array : empty;
    import std.algorithm : uniq;

    static immutable tok = tokenize(expr);
    // reduce to scalar
    static if (tok.output.empty) {
        static immutable inDim2Char = tok.inputs[0];
        alias x = xs[0];
        static foreach (i; 0 .. S.length) {
            static assert(tok.inputs[i].length == DimensionCount!(S[i]));
        }

        // create merged char list and lengths over arguments
        DeepElementType!S ret = 0;
        auto lengths = x.indexLength!inDim2Char;
        size_t[char] idx;
        mixin({
                string ret;
                foreach (c; inDim2Char.uniq) {
                    ret ~= format!`
foreach (%s; 0 .. lengths['%s']) {
idx['%s'] = %s;
`(c, c, c, c);
                }
                ret ~= "typeof(ret) tmp = 1;\n";
                ret ~= "static foreach (_i; 0 .. S.length) {\n";
                ret ~= "tmp *= xs[_i][dim2index!inDim2Char(idx).expand];\n";
                ret ~= "}\n";
                ret ~= "ret += tmp;\n";
                foreach (c; inDim2Char.uniq) {
                    ret ~= "}\n";
                }
                return ret;
            }());

        // NOTE: simplest example of generated code for 1-dim 1-arg
        // static foreach (ic; inDim2Char.uniq) {
        //     foreach (i0; 0 .. lengths[ic]) {
        //         idx[ic] = i0;
        //         ret += x[dim2index!inDim2Char(idx).expand];
        //     }
        // }
        return ret;
    } else {
        static assert(false, "not supported expr: " ~ expr);
    }
}

/// TODO support everything in this example
/// https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html

/// trace and scalar reduction
unittest
{
    auto m = iota(5, 5);
    writeln(m);
    assert(m.einsum!"ii" == 60);
    assert(m.einsum!"ij" == 300);
    assert(m.einsum!"ji" == 300);
}

void main()
{
    writeln("Edit source/app.d to start your project.");
}
