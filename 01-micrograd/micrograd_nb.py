import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from graphviz import Digraph

    # Very simple function
    # def f(x):
    #     return 3 * x**2 - 4 * x + 5


    # print(f(3.0))
    #
    # xs = np.arange(-5, 5, 0.25)
    # ys = f(xs)
    #
    # plt.plot(xs, ys)
    # plt.show()

    # Simple derivative estimation
    # h = 0.00000001
    # x = 2 / 3
    # print(f(x + h))
    #
    # print((f(x + h) - f(x)) / h)

    # More complex example
    a = 2.0
    b = -3.0
    c = 10.0
    d = a * b + c
    print(d)

    h = 0.000000001

    d_before = a * b + c
    c += h
    d_after = a * b + c

    print("d1:", d_before)
    print("d2:", d_after)
    print("slope:", (d_after - d_before) / h)


    class Value:
        """
        Stores a value, the values and operations it inherits from, to allow for autograd
        Children and operations are necessary so we can perform chain rule
        """

        def __init__(self, data, _children=(), _op="", label=""):
            self.data = data  # Set up on initial pass, saving numbers as the attribute
            self.grad = 0.0
            self._prev = set(
                _children  # Children allow us to track inheritance, necessary for autograd
            )
            self._op = _op  # Allows us to know the operation performed so we can take the derivative
            self.label = label

        def __repr__(self):  # so we have a nice string representation
            return f"Value(data={self.data})"

        def __add__(self, other):  # First pass only returns addition
            out = Value(
                self.data + other.data, (self, other), "+"
            )  # second pass returns children too
            return out

        def __mul__(self, other):  # First pass returns multiplication
            out = Value(
                self.data * other.data, (self, other), "*"
            )  # second pass returns children too
            return out

        def tanh(self):
            n = self.data
            t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
            out = Value(t, (self, ), 'tanh')
            return out


    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f
    print(d)
    print(d._prev)


    # Show a graph of all this
    def trace(root):
        # builds a set of all nodes and edges in a graph
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges


    def draw_dot(root):
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

        nodes, edges = trace(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(
                name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape="record"
            )
            if n._op:
                # check if it's an operation and create an op node for it
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot

    return L, Value, a, b, c, d, draw_dot, e, f, math, np, plt, random


@app.cell
def _(L, a, b, c, d, draw_dot, e, f):
    # Derivatives
    # We find all the derivatives by iteratively applying the chain rule and storing the gradient and then applying it at the next layer
    L.grad = 1.0

    # First level
    f.grad = 4.0
    d.grad = -2.0

    # Second level
    # dL/de = dL/dd * dd/de
    e.grad = d.grad * 1.0
    c.grad = d.grad * 1.0

    # Third level
    # dL/da = dL/de * de/da
    a.grad = e.grad * b.data
    b.grad = e.grad * a.data

    draw_dot(L)
    return


@app.cell
def _(Value):
    def lol():

        h = 0.0001
    
        a = Value(2.0, label="a")
        b = Value(-3.0, label="b")
        c = Value(10.0, label="c")
        e = a * b
        e.label = "e"
        d = e + c
        d.label = "d"
        f = Value(-2.0, label="f")
        L = d * f
        L.label = 'L'

        L1 = L.data

        a = Value(2.0, label="a")
        b = Value(-3.0 + h, label="b") # This is when we want dL/db, but we can do this with each variable and confirm that the calculations done by hand above using the chain rule match our results here.
        c = Value(10.0, label="c")
        e = a * b
        e.label = "e"
        d = e + c
        d.label = "d"
        f = Value(-2.0, label="f")
        L = d * f
        L.label = 'L'
        L2 = L.data

        print((L2 - L1) / h)

    lol()
    
    return


@app.cell
def _(L, a, b, c, d, e, f):
    # nudging values in the direction of the gradient should cause L to increase
    print(L)

    for point in [a, b, c, f]:
        point.data += 0.01 * point.grad

    e_new = a * b
    d_new = e + c
    L_new = d * f

    print(L_new.data)
    return (L_new,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(L, L_new, mo):
    mo.md(rf"""
    As we can see, $L$ has increased by ${L_new.data - L.data:.2f}$.
    """)
    return


@app.cell
def _(np, plt):
    plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2)))
    plt.grid()
    plt.show()
    return


@app.cell
def _(Value, draw_dot):
    # inputs x1 and x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights w1 and w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # Bias
    bias = Value(6.8813735890195432, label='b')

    # Steps broken up so we get pointers to the intermediate nodes
    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + bias; n.label = 'n'
    o = n.tanh()
    o.label = 'o'
    o.grad = 1.0

    draw_dot(o)
    return bias, n, o, w1, w2, x1, x1w1, x1w1x2w2, x2, x2w2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we calculate derivatives! First, we note that if $o = \tanh(n)$, then $\frac{d o}{d n} = 1 - tanh^2(n) = 1 - o^2.$
    """)
    return


@app.cell
def _(bias, draw_dot, n, o, w1, w2, x1, x1w1, x1w1x2w2, x2, x2w2):
    n.grad = 1 - o.data**2

    # do/db = do/dn * dn/db
    bias.grad = n.grad * 1.0
    x1w1x2w2.grad = n.grad * 1.0
    x1w1.grad = x1w1x2w2.grad * 1.0
    x2w2.grad = x1w1x2w2.grad * 1.0
    w1.grad = x1w1.grad * x1.data
    w2.grad = x2w2.grad * x2.data
    x1.grad = x1w1.grad * w1.data
    x2.grad = x2w2.grad * w2.data

    draw_dot(o)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alright, now, defining these gradients by hand is way too tedious and not realistic. We'll remake the `Value` class from earlier, but building in the automatic differentiation into it.
    """)
    return


@app.cell
def _(math):
    class ValueWithGrad:
        """
        Stores a value, the values and operations it inherits from, to allow for autograd
        Children and operations are necessary so we can perform chain rule
        """

        def __init__(self, data, _children=(), _op="", label=""):
            self.data = data
            self.grad = 0.0
            self._backward = lambda: None # This is new
            self._prev = set(
                _children
            )
            self._op = _op
            self.label = label

        def __repr__(self): 
            return f"ValueWithGrad(data={self.data})"

        def __add__(self, other): 
            other = other if isinstance(other, ValueWithGrad) else ValueWithGrad(other)
            out = ValueWithGrad(
                self.data + other.data, (self, other), "+"
            )
            def _backward():
                self.grad += 1.0 * out.grad
                other.grad += 1.0 * out.grad
            out._backward = _backward
            return out

        def __radd__(self, other):
            return self + other

        def __mul__(self, other):
            other = other if isinstance(other, ValueWithGrad) else ValueWithGrad(other)
            out = ValueWithGrad(
                self.data * other.data, (self, other), "*"
            )
            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _backward
            return out

        def __rmul__(self, other):
            return self * other

        def __pow__(self, other):
            assert isinstance(other, (int, float)), "only supporting float/int now"
            out = ValueWithGrad(self.data ** other, (self, ), f'**{other}')

            def _backward():
                self.grad += other * self.data ** (other - 1) * out.grad

            out._backward = _backward
            return out

        def __truediv__(self, other):
            return self.__mul__(other**-1)

        def __neg__(self):
            return self.__mul__(-1)

        def __sub__(self, other):
            return self + (-other)

        def __rsub__(self, other):
            return self - other

        def tanh(self):
            n = self.data
            t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
            out = ValueWithGrad(t, (self, ), 'tanh')
            def _backward():
                self.grad += (1 - t**2) * out.grad
            out._backward = _backward
            return out

        def exp(self):
            n = self.data
            out = ValueWithGrad(math.exp(n), (self, ), 'exp')

            def _backward():
                self.grad += out.data * out.grad

            out._backward = _backward

            return out

        def backward(self):
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)

            build_topo(self)

            self.grad = 1.0
            for node in reversed(topo):
                node._backward()

    return (ValueWithGrad,)


@app.cell
def _(ValueWithGrad, draw_dot):
    # inputs x1 and x2
    x1_new = ValueWithGrad(2.0, label='x1')
    x2_new = ValueWithGrad(0.0, label='x2')

    # weights w1 and w2
    w1_new = ValueWithGrad(-3.0, label='w1')
    w2_new = ValueWithGrad(1.0, label='w2')

    # Bias
    bias_new = ValueWithGrad(6.8813735890195432, label='b')

    # Steps broken up so we get pointers to the intermediate nodes
    x1w1_new = x1_new * w1_new; x1w1_new.label = 'x1*w1'
    x2w2_new = x2_new * w2_new; x2w2_new.label = 'x2*w2'

    x1w1x2w2_new = x1w1_new + x2w2_new; x1w1x2w2_new.label = 'x1*w1 + x2*w2'
    n_new = x1w1x2w2_new + bias_new; n_new.label = 'n'
    o_new = n_new.tanh()
    o_new.label = 'o'
    o_new.grad = 1.0

    draw_dot(o_new)
    return (o_new,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the root nodes need to be back propagated first, we need to run a topological sort on the nodes to get them in order:
    """)
    return


@app.cell
def _(o_new):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(o_new)
    topo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll now add the topological sorting back into the `ValueWithDiff` class so we can apply this in one backward pass.
    """)
    return


@app.cell
def _(draw_dot, o_new):
    o_new.backward()

    draw_dot(o_new)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have a bad bug! See this example.
    """)
    return


@app.cell
def _(ValueWithGrad, draw_dot):
    a_bad = ValueWithGrad(3.0, label='a')
    b_bad = a_bad + a_bad; b_bad.label = 'b'
    b_bad.backward()
    draw_dot(b_bad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is equivalent to saying $b = 2a$ so it should be clear that the gradient is $2$, though here it is saying that it's one.

    Another example, also showing how using a variable multiple times leads to issues.
    """)
    return


@app.cell
def _(ValueWithGrad, draw_dot):
    a2 = ValueWithGrad(-2.0, label='a')
    b2 = ValueWithGrad(3.0, label='b')
    d2 = a2 * b2; d2.label = 'd'
    e2 = a2 + b2; e2.label = 'e'
    f2 = d2 * e2; f2.label = 'f'

    f2.backward()

    draw_dot(f2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The issue here is that the first pass propagates the gradient from $d$ into $a$ and $b$, and then the next step propagates the gradient from $e$ into $a$ and $b$, overwriting the existing gradient.

    We can fix this by just adding the gradients. I'll modify the code above from
    ```python
    self.grad = grad_calculation
    ```
    to
    ```python
    self.grad += grad_calculation
    ```
    instead.

    Since I'm working in a notebook, only the latest version of the code will be visible.
    """)
    return


@app.cell
def _(ValueWithGrad):
    ValueWithGrad(2) + 4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Neural Nets

    Now that we've got our autograd framework set up alright, we're going to move forwards into defining neural network classes from scratch as well.
    """)
    return


@app.cell
def _(ValueWithGrad, random):
    class Neuron:
        def __init__(self, nin):
            self.w = [ValueWithGrad(random.uniform(-1, 1)) for _ in range(nin)]
            self.b = ValueWithGrad(random.uniform(-1, 1))

        def __call__(self, x):
            # w * x + b
            act = sum((wi * xi for (wi, xi) in zip(self.w, x)), self.b)
            out = act.tanh()
            return out

        def parameters(self):
            return self.w + [self.b]

    class Layer:
        def __init__(self, nin, nout):
            self.neurons = [Neuron(nin) for _ in range(nout)]

        def __call__(self, x):
            outs = [n(x) for n in self.neurons]
            return outs[0] if len(outs) == 1 else outs

        def parameters(self):
            return [p for neuron in self.neurons for p in neuron.parameters()]

    class MLP:
        def __init__(self, nin, nouts):
            sz = [nin] + nouts
            self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def parameters(self):
            return [p for layer in self.layers for p in layer.parameters()]

    def test():
        x = [2.0, 3.0, -1.0]
        n = MLP(3, [4, 4, 1])
        return n(x)

    test()
    return MLP, test


@app.cell
def _(draw_dot, test):
    draw_dot(test())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's try training a neural net.
    """)
    return


@app.cell
def _(MLP):
    xs = [
        [3.0, 1.0, -1.5],
        [-1, 2.0, 2.5],
        [0.5, 2.5, -1.0],
        [1.0, -1.0, -0.5],
    ]
    ys = [-1.0, 1.0, 1.0, -1.0]
    mlp = MLP(3, [4, 4, 1])
    ypred = [mlp(x) for x in xs]
    print(ypred)
    return mlp, xs, ypred, ys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The "loss" is a measure of our neural network's performance. Here, we'll use MSE as a loss.
    """)
    return


@app.cell
def _(ypred, ys):
    loss = sum((yout - ygt)**2 for ygt, yout, in zip(ys, ypred))
    print(loss)
    return (loss,)


@app.cell
def _(loss):
    loss.backward()
    return


@app.cell
def _(mlp):
    mlp.layers[0].neurons[0].w[0].grad
    return


@app.cell
def _(draw_dot, loss):
    draw_dot(loss)
    return


@app.cell
def _(mlp):
    for p in mlp.parameters():
        p.data -= 0.01 * p.grad
    return


@app.cell
def _(mlp, xs, ys):
    ypred1 = [mlp(x) for x in xs]
    loss1 = sum((yout - ygt)**2 for ygt, yout, in zip(ys, ypred1))
    loss1
    return


@app.cell
def _(mlp, plt, xs, ys):

    loss_vals = []
    steps = 100
    for i in range(steps):
        # Forward pass
        preds = [mlp(x) for x in xs]
        new_loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, preds))

        # Backward pass
        for par in mlp.parameters():
            par.grad = 0.0
        new_loss.backward()

        # Update
        for par in mlp.parameters():
            par.data -= 0.01 * par.grad

        # Show results
        loss_vals.append(new_loss.data)

    plt.plot(range(1, steps + 1), loss_vals)
    plt.xlabel("Training step")
    plt.ylabel("Loss (MSE)")
    return new_loss, preds


@app.cell
def _(preds):
    preds
    return


@app.cell
def _(new_loss):
    new_loss.data
    return


if __name__ == "__main__":
    app.run()
