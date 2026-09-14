import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flysoc.brain import BrainLab, schematic_positions
from flysoc.brain_server import make_handler
from flysoc.connectome import ConnectomePulse, parse_positions
from flysoc.ingestion import chronological_split
from flysoc.pipeline import FlySOCPipeline


def test_position_parsing_never_evaluates_data():
    np.testing.assert_array_equal(parse_positions(pd.Series(['[1 2 3]', '[4 5 6]'])), [[1,2,3],[4,5,6]])
    with pytest.raises(ValueError):
        parse_positions(pd.Series(['__import__("os").getcwd()']))
    with pytest.raises(ValueError):
        parse_positions(pd.Series(['[1 nan 3]']))


def test_graph_pulse_matches_literal_recurrence_and_mass_decay():
    transition = sparse.csr_matrix([[0,1,0],[0,0,.5],[0,0,0]], dtype=np.float32)
    pulse = ConnectomePulse(transition)
    pulse.reset(np.array([0]))
    first = pulse.step()
    np.testing.assert_allclose(first, [.18,.72,0], atol=1e-6)
    np.testing.assert_allclose(pulse.step(), [.0324,.2592,.2592], atol=1e-6)
    assert pulse.activity.sum() < first.sum()
    for _ in range(20):
        previous = pulse.activity.sum()
        assert pulse.step().sum() <= previous + 1e-7
    pulse.reset(np.array([], dtype=int))
    assert not pulse.step().any()


def test_pulse_rejects_unstable_or_negative_graph():
    with pytest.raises(ValueError, match='substochastic'):
        ConnectomePulse(sparse.csr_matrix([[2]]))
    with pytest.raises(ValueError, match='nonnegative'):
        ConnectomePulse(sparse.csr_matrix([[-1]]))
    pulse = ConnectomePulse(sparse.eye(2))
    with pytest.raises(ValueError):
        pulse.reset(np.array([2]))


def test_live_trace_equals_saved_algorithm(config, alerts):
    splits = chronological_split(alerts)
    pipeline = FlySOCPipeline(config).fit(splits['train'])
    lab = BrainLab.__new__(BrainLab)
    lab.pipeline = pipeline
    lab.test = splits['test']
    lab.payload = {'classifiers': {}}
    trace = lab.trace_alert(0)
    pn = pipeline.preprocessor.transform(lab.test.iloc[[0]])
    expected_fp = pipeline.encode(lab.test.iloc[[0]])
    np.testing.assert_array_equal(trace['winner_indices'], expected_fp.indices)
    np.testing.assert_allclose(trace['kc_activations'], (pn @ pipeline.encoder.projection_).toarray()[0])
    edges = np.asarray(trace['contributing_edges']).reshape(-1,2)
    assert all(pipeline.encoder.projection_[pre,post] == 1 for pre,post in edges)
    assert set(edges[:,0]).issubset(set(pn.indices))
    assert set(edges[:,1]).issubset(set(expected_fp.indices))
    assert trace['active_kcs'] == 16
    with pytest.raises(ValueError):
        lab.trace_alert(-1)
    with pytest.raises(ValueError, match='flat JSON'):
        lab.trace_alert(alert={'nested': {'invalid': 'structure'}})


def test_schematic_layout_is_reproducible_and_labeled_separately():
    a = schematic_positions(100, 'pn')
    np.testing.assert_array_equal(a, schematic_positions(100, 'pn'))
    assert a.shape == (100,3) and np.isfinite(a).all()


def test_loopback_server_origin_and_body_checks():
    class FakeLab:
        connectome_summary = {'neurons': 1}
        nodes = pd.DataFrame([{'root_id': '720575940629663000', 'class': 'ALPN',
                               'super_class': 'central', 'nt_type': 'ACH'}])
        def status(self): return {'mode':'local'}
        def trace_alert(self, index, alert): return {'index': index}
    server = ThreadingHTTPServer(('127.0.0.1',0), make_handler(FakeLab()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(url+'/api/status') as response:
            assert json.load(response)['mode'] == 'local'
        with urllib.request.urlopen(url+'/api/connectome/neuron/0') as response:
            assert json.load(response)['root_id'] == '720575940629663000'
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url+'/api/connectome/neuron/-1')
        assert error.value.code == 400
        data = json.dumps({'index':2}).encode()
        request = urllib.request.Request(url+'/api/fly/trace', data=data, headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)['index'] == 2
        foreign = urllib.request.Request(url+'/api/fly/trace', data=data, headers={'Content-Type':'application/json','Origin':'https://foreign.example'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(foreign)
        assert error.value.code == 403
        request = urllib.request.Request(url+'/api/status', headers={'Host':'foreign.example'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url+'/../config/default.yaml')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
