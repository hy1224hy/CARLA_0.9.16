"""Disjoint scenario seed namespaces. A/B gate cases are strictly paired."""
FAMILIES = ("parallel", "closing")
OFFSETS = {"train": 0, "gate": 1_000_000, "selection": 2_000_000, "test": 3_000_000}


def cases(split, per_family):
    if type(per_family) is not int or not 1 <= per_family <= 100000:
        raise ValueError("per_family must be an integer in [1,100000]")
    return [(family, OFFSETS[split] + f * 100_000 + i)
            for f, family in enumerate(FAMILIES) for i in range(per_family)]


def starts(rng, family):
    if family not in FAMILIES:
        raise ValueError(f"unknown scenario family {family}")
    # IDs 0,1 red IDM; IDs 2,3 blue learners. One pair in each lane.
    red_x = 70 + rng.uniform(-3, 3, 2)
    gap = rng.uniform(14, 22, 2)
    blue_x = red_x + gap if family == "closing" else red_x - gap
    red_speed = 17.0 if family == "closing" else 15.0
    blue_speed = 12.0 if family == "closing" else 17.0
    return [(red_x[i], i * 4.0, red_speed) for i in range(2)] + [
        (blue_x[i], i * 4.0, blue_speed) for i in range(2)]
