class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.components])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


if __name__ == "__main__":
    print("=== Vectors ===")
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * 3 = {a * 3}")
    print(f"a · b = {a.dot(b)}")
    print(f"|a| = {a.magnitude():.4f}")
    print(f"â (normalized) = {a.normalize()}")
    print(f"cosine_similarity(a, b) = {a.cosine_similarity(b):.4f}")

    print("\n=== Matrices ===")
    rotation_90 = Matrix([[0, -1], [1, 0]])
    point = Vector([3, 1])
    rotated = rotation_90 @ point
    print(f"Rotate {point} by 90° → {rotated}")

    print("\n=== Neural Network Layer (Matrix × Vector) ===")
    import random
    random.seed(42)
    weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
    input_vec = Vector([1.0, 0.5, -0.3])
    output = weights @ input_vec
    print(f"Input (3D):  {input_vec}")
    print(f"Output (2D): {output}")
    print("^ This is literally what a neural network layer does.")
