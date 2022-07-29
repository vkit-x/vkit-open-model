from vkit_open_model.evaluation.opt import pad_length_to_make_divisible


def test_pad_length_to_make_divisible():
    padded_length, pad = pad_length_to_make_divisible(6, 3)
    assert padded_length == 6
    assert pad == 0

    padded_length, pad = pad_length_to_make_divisible(7, 3)
    assert padded_length == 9
    assert pad == 2
