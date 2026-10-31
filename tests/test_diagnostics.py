import numpy as np
import pytest
from unsgrpo.diagnostics import StableMapper, aggregate, offset_consensus, rank_agreement, student_loss


def test_mapper_constant_and_cross_boundary():
    mapper = StableMapper([[0, 2], [2, 4], [4, 4.5]], [[0, 3], [3, 4.5]])
    assert mapper([10, 40, 100]) == pytest.approx([20, 60])
    assert np.array_equal(mapper([.1234567]*3), np.array([.1234567]*2))


def test_mapper_rejects_gap():
    with pytest.raises(ValueError, match="coverage"):
        StableMapper([[0, 1], [2, 3]], [[0, 3]])


def test_overlap_consensus_has_no_effect_on_consistent_predictions():
    ids = [[0, 1, 2], [1, 2, 3]]
    values = [[.1, .2, .3], [.2, .3, .4]]
    corrected, b = offset_consensus(ids, values)
    assert np.array_equal(b, [0, 0])
    assert aggregate(4, ids, corrected) == pytest.approx([.1, .2, .3, .4])


def test_overlap_consensus_reduces_injected_window_offset():
    ids = [[0, 1, 2], [1, 2, 3]]
    values = [[.3, .4, .5], [.0, .1, .2]]
    corrected, b = offset_consensus(ids, values, ridge=1)
    assert b.sum() == pytest.approx(0)
    assert abs(corrected[0][1]-corrected[1][0]) < .4
    assert b[0] > 0 and b[1] < 0


def test_rank_gradient_pushes_inverted_pair_in_correct_direction():
    import torch
    pred = torch.tensor([[.8, .2]], requires_grad=True)
    q = torch.tensor([[.2, .8]])
    loss = student_loss(pred, q, q, "rank_cal")
    loss.backward()
    assert pred.grad[0, 0] > 0 and pred.grad[0, 1] < 0
    assert rank_agreement([.2, .8], [.2, .8]) == 1
    assert rank_agreement([.8, .2], [.2, .8]) == 0
    assert rank_agreement([.5, .5], [.2, .8]) == .5


def test_all_teacher_ties_have_finite_loss():
    import torch
    p = torch.tensor([[.3, .7]], requires_grad=True)
    q = torch.tensor([[.5, .5]])
    loss = student_loss(p, q, q, "rank_cal")
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
