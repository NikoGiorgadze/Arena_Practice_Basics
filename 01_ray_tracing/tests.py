import torch as t


def make_test_rays_1d(num_pixels: int = 9, y_limit: float = 10.0) -> t.Tensor:
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    return rays


rays1d = make_test_rays_1d()
segments = t.tensor(
    [
        [[1.0, -12.0, 0.0], [1.0, -6.0, 0.0]],
        [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]],
        [[2.0, 12.0, 0.0], [2.0, 21.0, 0.0]],
    ]
)


def test_intersect_ray_1d(intersect_ray_1d):
    expected = [(0, 0), (0, 1), (2, 7), (2, 8)]
    actual = []
    for i, segment in enumerate(segments):
        for j, ray in enumerate(rays1d):
            if intersect_ray_1d(ray, segment):
                actual.append((i, j))
    if expected != actual:
        print("Expected segment-ray intersections: ", expected)
        print("Actual:", actual)
    assert expected == actual
    print("All tests in `test_intersect_ray_1d` passed!")


def test_intersect_ray_1d_special_case(intersect_ray_1d):
    ray = t.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    segment = t.tensor([[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]])
    actual = intersect_ray_1d(ray, segment)
    expected = False
    assert actual == expected
    print("All tests in `test_intersect_ray_1d_special_case` passed!")


def test_intersect_rays_1d(intersect_rays_1d):
    expected = t.tensor([True, True, False, False, False, False, False, True, True])
    actual = intersect_rays_1d(rays1d, segments)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_intersect_rays_1d` passed!")


def test_intersect_rays_1d_special_case(intersect_rays_1d):
    ray = t.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, -10.0, 0.0]],
        ]
    )
    segment = t.tensor(
        [
            [[0.0, 2.0, 2.0], [0.0, 4.0, 0.0]],
            [[1.0, -12.0, 0.0], [1.0, -6.0, 0.0]],
        ]
    )
    actual = intersect_rays_1d(ray, segment)
    expected = t.tensor([False, True])
    t.testing.assert_close(actual, expected)
    print("All tests in `test_intersect_rays_1d_special_case` passed!")


def test_triangle_ray_intersects(triangle_ray_intersects):
    A = t.tensor([2, 0.0, -1.0])
    B = t.tensor([2, -1.0, 0.0])
    C = t.tensor([2, 1.0, 1.0])
    rays = t.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0000, 0.3333, 0.3333]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, -1.0]],
            [[0.0, 0.0, 0.0], [-1.000, -0.3333, -0.3333]],
        ]
    )
    expected = [True, False, False]
    for (O, D), expected_i in zip(rays, expected):
        actual = triangle_ray_intersects(A, B, C, O, D)
        assert actual == expected_i
    print("All tests in `test_triangle_ray_intersects` passed!")


def test_rotation_matrix(rotation_matrix):
    id = rotation_matrix(t.tensor(0.0))
    id_expected = t.eye(3)
    t.testing.assert_close(id, id_expected)

    ninety = rotation_matrix(t.tensor(t.pi / 2))
    ninety_expected = t.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    t.testing.assert_close(ninety, ninety_expected)

    rotated = rotation_matrix(theta := t.tensor(0.1))
    rotated_expected = t.tensor(
        [
            [t.cos(theta), 0.0, t.sin(theta)],
            [0.0, 1.0, 0.0],
            [-t.sin(theta), 0.0, t.cos(theta)],
        ]
    )
    t.testing.assert_close(rotated, rotated_expected)

    print("All tests in `test_rotation_matrix` passed!")
