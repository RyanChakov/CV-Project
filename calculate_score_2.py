def get_score_from_radius(hit_radius):

    max_radius = 100

    scoring_ranges = [
        (0.9 * max_radius, 1.0000001 * max_radius),
        (0.8 * max_radius, 0.9 * max_radius),
        (0.69 * max_radius, 0.80 * max_radius),
        (0.59 * max_radius, 0.69 * max_radius),
        (0.49 * max_radius, 0.59 * max_radius),
        (0.39 * max_radius, 0.49 * max_radius),
        (0.29 * max_radius, 0.39 * max_radius),
        (0.189 * max_radius, 0.29 * max_radius),
        (0.085 * max_radius, 0.189 * max_radius)
    ]

    bullseye_score = 10

    for i, (lower_bound, upper_bound) in enumerate(scoring_ranges):
        if lower_bound <= hit_radius < upper_bound:
            return i + 1
    if 0 < hit_radius <= 0.085 * max_radius:
        return bullseye_score

    return 0

score = get_score_from_radius(hit_radius=0)

print(score)