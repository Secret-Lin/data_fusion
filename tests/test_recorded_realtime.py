import time

import pytest

from tools.benchmark_recorded_realtime import LoopbackReceiver, receiver_metrics, statistics


def test_repeated_packets_are_not_fresh_states():
    rows = [{'received': .1 + i / 30, 'sent': .1 + i / 30 - .001,
             'capture_due': 0., 'update_id': 1,
             'fused': {'object_name': 'can', 'sync_group': 4}} for i in range(27)]
    result = receiver_metrics(rows, 0., 1., 30.)
    assert result['packet_hz'] == 27
    assert result['distinct_update_hz'] == 1
    assert result['distinct_capture_group_hz'] == 1
    assert result['repeat_fraction'] == pytest.approx(26 / 27)
    assert result['result_age_ms']['p95'] > 900
    assert result['fraction_age_within_one_receiver_period'] == 0


def test_revision_is_new_update_not_new_capture_group():
    rows = [{'received': .1 * i, 'sent': .1 * i, 'capture_due': 0., 'update_id': i,
             'fused': {'object_name': 'can', 'sync_group': 4}} for i in range(3)]
    result = receiver_metrics(rows, 0., 1., 30.)
    assert result['distinct_update_hz'] == 3
    assert result['distinct_capture_group_hz'] == 1


def test_no_output_not_zero_latency():
    assert statistics([])['p95'] is None
    result = receiver_metrics([], 0., 1., 30.)
    assert result['result_age_ms']['p95'] is None
    assert result['repeat_fraction'] is None


def test_loopback_threads_preserve_payload_and_monotonic_clock():
    receiver = LoopbackReceiver(30.)
    start = time.perf_counter()
    try:
        receiver.start()
        receiver.update({'object_name': 'can', 'sync_group': 3, 'T': [[1.]]}, start, start, 0)
        time.sleep(.2)
    finally:
        receiver.close()
    assert not receiver.errors
    assert len(receiver.rows) >= 2
    assert all(r['received'] >= r['sent'] >= r['published'] for r in receiver.rows)
    assert all(r['fused']['T'] == [[1.]] for r in receiver.rows)
    assert len({r['update_id'] for r in receiver.rows}) == 1


def test_invalid_rate():
    with pytest.raises(ValueError):
        LoopbackReceiver(0.)
