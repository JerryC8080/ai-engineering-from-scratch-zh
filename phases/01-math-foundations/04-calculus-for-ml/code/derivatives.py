import math
import random


def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient


def gradient_descent_1d(f, df, x0, lr=0.1, steps=20):
    x = x0
    history = []
    for step in range(steps):
        grad = df(x)
        x = x - lr * grad
        history.append((step, x, f(x)))
    return x, history


def gradient_descent_nd(f, x0, lr=0.1, steps=100):
    point = list(x0)
    history = []
    for step in range(steps):
        grad = numerical_gradient(f, point)
        point = [p - lr * g for p, g in zip(point, grad)]
        history.append((step, list(point), f(point)))
    return point, history


def demo_numerical_vs_analytical():
    print("=" * 55)
    print("NUMERICAL vs ANALYTICAL DERIVATIVES")
    print("=" * 55)

    test_cases = [
        ("x^2",    lambda x: x**2,        lambda x: 2*x),
        ("x^3",    lambda x: x**3,        lambda x: 3*x**2),
        ("sin(x)", lambda x: math.sin(x), lambda x: math.cos(x)),
        ("e^x",    lambda x: math.exp(x), lambda x: math.exp(x)),
        ("1/x",    lambda x: 1/x,         lambda x: -1/x**2),
    ]

    x = 2.0
    print(f"\nAt x = {x}:")
    print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
    print("-" * 50)
    for name, f, df in test_cases:
        num = numerical_derivative(f, x)
        ana = df(x)
        err = abs(num - ana)
        print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")


def demo_gradient():
    print("\n" + "=" * 55)
    print("GRADIENT (VECTOR OF PARTIAL DERIVATIVES)")
    print("=" * 55)

    def f(point):
        x, y = point
        return x**2 + 3*x*y + y**2

    point = [1.0, 2.0]
    grad = numerical_gradient(f, point)
    analytical = [2*point[0] + 3*point[1], 3*point[0] + 2*point[1]]

    print(f"\nf(x,y) = x^2 + 3xy + y^2")
    print(f"At point ({point[0]}, {point[1]}):")
    print(f"  Numerical gradient:  [{grad[0]:.4f}, {grad[1]:.4f}]")
    print(f"  Analytical gradient: [{analytical[0]:.1f}, {analytical[1]:.1f}]")


def demo_gradient_descent_1d():
    print("\n" + "=" * 55)
    print("GRADIENT DESCENT: f(x) = x^2")
    print("=" * 55)

    x = 5.0
    lr = 0.1
    print(f"\nStart: x={x}, lr={lr}")
    for step in range(20):
        grad = 2 * x
        x = x - lr * grad
        if step % 4 == 0 or step == 19:
            print(f"  step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
    print(f"Minimum found at x={x:.6f} (true minimum: x=0)")


def demo_gradient_descent_2d():
    print("\n" + "=" * 55)
    print("GRADIENT DESCENT: f(x,y) = x^2 + y^2")
    print("=" * 55)

    def f(point):
        x, y = point
        return x**2 + y**2

    point = [4.0, 3.0]
    lr = 0.1
    print(f"\nStart: ({point[0]}, {point[1]}), lr={lr}")
    for step in range(30):
        grad = numerical_gradient(f, point)
        point = [p - lr * g for p, g in zip(point, grad)]
        loss = f(point)
        if step % 5 == 0 or step == 29:
            print(f"  step {step:2d}  ({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
    print(f"Minimum found at ({point[0]:.4f}, {point[1]:.4f}) (true: (0, 0))")


def demo_linear_regression():
    print("\n" + "=" * 55)
    print("GRADIENT DESCENT: LINEAR REGRESSION y = 2x + 1")
    print("=" * 55)

    random.seed(42)
    w = random.gauss(0, 1)
    b = random.gauss(0, 1)
    lr = 0.01

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 5.0, 7.0, 9.0, 11.0]

    for epoch in range(200):
        total_loss = 0
        dw = 0
        db = 0
        for x, y in zip(xs, ys):
            pred = w * x + b
            error = pred - y
            total_loss += error ** 2
            dw += 2 * error * x
            db += 2 * error
        dw /= len(xs)
        db /= len(xs)
        total_loss /= len(xs)
        w -= lr * dw
        b -= lr * db
        if epoch % 40 == 0 or epoch == 199:
            print(f"  epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

    print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
    print(f"Actual:  y = 2.00x + 1.00")


if __name__ == "__main__":
    demo_numerical_vs_analytical()
    demo_gradient()
    demo_gradient_descent_1d()
    demo_gradient_descent_2d()
    demo_linear_regression()
