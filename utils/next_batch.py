import math


def next_batch(views, batch_size):
    tot = views[0].shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x = []
        for j in range(len(views)):
            batch_x.append(views[j][start_idx: end_idx, ...])

        yield (batch_x, (i + 1))
