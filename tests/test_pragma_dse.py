import dataclasses

import pytest

from resnet8_hls.model import describe_network
from resnet8_hls.pragma_dse import (
    format_pragma_summary,
    generate_pragma_set,
    parallel_macs,
    safe_unroll,
)

from .conftest import build_random_resnet


@pytest.mark.parametrize(
    ("bound", "requested", "expected"),
    [
        (3, 1, 1),  # sequential stays sequential
        (3, 0, 0),  # complete unroll stays complete
        (3, 3, 0),  # covering the whole loop means complete unroll
        (3, 5, 0),
        (16, 2, 2),
        (12, 5, 4),  # largest divisor of 12 that is <= 5
        (7, 4, 1),  # prime bound: only the trivial divisor fits
    ],
)
def test_safe_unroll(bound, requested, expected):
    assert safe_unroll(bound, requested) == expected


def test_parallel_macs():
    assert parallel_macs(16, 1, 1, 1) == 1
    assert parallel_macs(16, 0, 1, 1) == 16
    assert parallel_macs(16, 2, 0, 0) == 2 * 3 * 3


@pytest.fixture
def pragma_set(tiny_state_dict):
    return generate_pragma_set(describe_network(tiny_state_dict))


def test_there_are_eight_ordered_configurations(pragma_set):
    assert len(pragma_set) == 8
    assert [cfg.label[:2] for cfg in pragma_set] == [f"P{i}" for i in range(8)]
    assert not any(cfg.use_dataflow for cfg in pragma_set)


def test_initiation_intervals(pragma_set):
    p = pragma_set
    assert (p[0].conv1.ii, p[0].resblock.ii, p[0].fc.ii) == (8, 8, 8)
    assert (p[1].conv1.ii, p[1].resblock.ii, p[1].fc.ii) == (4, 4, 4)
    assert (p[2].conv1.ii, p[2].resblock.ii, p[2].fc.ii) == (2, 2, 2)
    assert (p[3].conv1.ii, p[3].resblock.ii, p[3].fc.ii) == (2, 4, 4)
    assert (p[6].conv1.ii, p[6].resblock.ii, p[6].fc.ii) == (8, 4, 2)


def test_only_p7_unrolls_the_residual_blocks_and_it_partitions_the_weights(pragma_set):
    for cfg in pragma_set[:7]:
        assert cfg.resblock.unroll_ic == 1
        assert cfg.resblock.partition_w_factor == 1
    p7 = pragma_set[7]
    assert p7.resblock.unroll_ic == 2
    assert p7.resblock.partition_w_factor == 0  # complete partition
    assert p7.warning_flags == ()


def test_fc_unroll_only_in_p5(pragma_set):
    assert [cfg.fc.unroll_i for cfg in pragma_set] == [1, 1, 1, 1, 1, 2, 1, 1]


def test_estimates_follow_the_expected_ordering(pragma_set):
    p = pragma_set
    assert p[0].est_latency_k > p[1].est_latency_k > p[2].est_latency_k
    assert p[0].est_dsp == p[1].est_dsp == p[2].est_dsp  # no unrolling
    assert p[7].est_dsp > p[1].est_dsp  # IC unroll adds MAC units
    assert p[5].est_dsp == p[4].est_dsp + 1  # FC unroll 2 instead of 1
    assert all(cfg.est_lut_k > 0 for cfg in p)


def test_unrolling_without_partition_would_be_flagged(pragma_set):
    bad = dataclasses.replace(
        pragma_set[7], resblock=dataclasses.replace(pragma_set[7].resblock, partition_w_factor=1)
    )
    from resnet8_hls.pragma_dse import check_warnings

    assert check_warnings(bad) == ("W-MEMPORT [resblock] IC unroll active but no weight partition",)


def test_p7_falls_back_to_sequential_when_channels_are_tiny():
    model = build_random_resnet(stem=2, planes=(2, 2, 2))
    spec = describe_network({k: v.numpy() for k, v in model.state_dict().items()})
    p7 = generate_pragma_set(spec)[7]
    assert p7.resblock.unroll_ic == 1
    assert p7.resblock.partition_w_factor == 1


def test_summary_lists_every_configuration(pragma_set):
    text = format_pragma_summary(pragma_set)
    assert "8 configurations" in text
    assert all(cfg.label in text for cfg in pragma_set)
