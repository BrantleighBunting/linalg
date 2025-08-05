#!/usr/bin/python3
# Copyright (C) 2025 Brantleigh Bunting
# SPDX-License-Identifier: BSL-1.1
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file and at <https://mariadb.com/bsl11>

"""
Vector operations in python
"""
import math
import unittest


class Vector:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.z})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            raise Exception(f"Invalid type, got: {type(other)}")

        return (self.x, self.y, self.z) == (other.x, other.y, other.z)


def vec_add(u: Vector, v: Vector) -> Vector:
    return Vector(
        x=u.x + v.x,
        y=u.y + v.y,
        z=u.z + v.z,
    )


def scalar_mul(s: int, v: Vector) -> Vector:
    return Vector(
        x=s * v.x,
        y=s * v.y,
        z=s * v.z,
    )


def dot_product(u, v) -> float:
    """
    Implements the scalar (dot) product between two vectors.
    """
    return (u.x * v.x) + (u.y * v.y) + (u.z * v.z)


def cross_product(u, v) -> Vector:
    """
    Implements classical cross product u x v in R^3
    Defines a vector orthogonal to u and v with magnitude
    equal to the parallelogram area.
    """
    return Vector(
        x=u.y * v.z - u.z * v.y, y=u.z * v.x - u.x * v.z, z=u.x * v.y - u.y * v.x
    )


def length(u: Vector) -> float:
    return math.sqrt(pow(u.x, 2) + pow(u.y, 2) + pow(u.z, 2))


def angle(u, v):
    u_len = length(u)
    v_len = length(v)
    if u_len == 0 or v_len == 0:
        raise ValueError("Angle undefined for zero-length vector")

    uv_dot = dot_product(u, v)
    cos_theta = uv_dot / (u_len * v_len)
    # clamp angle radians between [-1, 1]
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def projection(direction, v: Vector):
    pass


def cosine_similarity(u: Vector, v: Vector) -> float:
    cos_theta = dot_product(u, v) / (length(u) * length(v))
    return max(-1.0, min(1.0, cos_theta))


class VectorTests(unittest.TestCase):
    def test_vec_add(self):
        self.assertEqual(Vector(5, 5, 5), vec_add(Vector(1, 1, 1), Vector(4, 4, 4)))

    def test_scalar_mul(self):
        self.assertEqual(Vector(10, 10, 10), scalar_mul(5, Vector(2, 2, 2)))

    def test_dot_product(self):
        self.assertEqual(75, dot_product(Vector(5, 5, 5), Vector(5, 5, 5)))

    def test_length(self):
        # Test with perfect square trinomials
        a = Vector(4, 0, 0)
        b = Vector(0, 4, 0)
        c = Vector(0, 0, 4)
        self.assertEqual(4, length(a))
        self.assertEqual(4, length(b))
        self.assertEqual(4, length(c))
        self.assertAlmostEqual(1.7320508076, length(Vector(1, 1, 1)))

    def test_cross_product(self):
        self.assertEqual(
            Vector(-10, 4, 8), cross_product(Vector(2, -1, 3), Vector(0, 4, -2))
        )
        # Canonical right-hand basis check, should result in vector along k hat
        u = Vector(1, 0, 0)
        v = Vector(0, 1, 0)
        self.assertEqual(Vector(0, 0, 1), cross_product(u, v))

        u = Vector(3, -3, 1)
        v = Vector(4, 9, 2)
        self.assertEqual(Vector(-15, -2, 39), cross_product(u, v))

        # Parallel vectors, result should be 0 vector
        u = Vector(2, 4, 6)
        v = Vector(1, 2, 3)
        self.assertEqual(Vector(0, 0, 0), cross_product(u, v))

        # Ensure multiplying by zero vector short circuits correctly
        u = Vector(0, 0, 0)
        v = Vector(5, -7, 1)
        self.assertEqual(Vector(0, 0, 0), cross_product(u, v))

    def test_angle(self):
        test_cases = [
            ((1, 0, 0), (0, 1, 0), math.pi / 2),
            ((1, 2, 3), (1, 2, 3), 0.0),
            ((1, 0, 0), (-1, 0, 0), math.pi),
            ((1, 0, 0), (1, 1, 0), math.pi / 4),
            ((2, -1, 3), (0, 4, -2), 2.21131864),
            ((123456, -98765, 50), (-23456, 8765, 100), 2.824433709487314),
            ((1e-8, 0, 0), (0, 1e-8, 0), math.pi / 2),
            ((3, -3, 1), (4, 9, 2), 1.8720947029995874),
        ]
        for u_vals, v_vals, expected in test_cases:
            u, v = Vector(*u_vals), Vector(*v_vals)
            self.assertAlmostEqual(angle(u, v), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
