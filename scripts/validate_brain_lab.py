"""Validate the downloaded graph and real-model traces; save measured evidence."""

import json
import os
from time import perf_counter

from flysoc.config import ROOT

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from flysoc.artifacts import environment_manifest, run_id, sha256, write_json
from flysoc.brain import BrainLab
from flysoc.connectome import CONNECTOME, ConnectomePulse


def main() -> None:
    output = ROOT / 'results' / run_id('brain_validation')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((CONNECTOME / 'download_manifest.json').read_text())
    for name, record in manifest['files'].items():
        assert sha256(CONNECTOME / 'raw' / name) == record['sha256'], name
    prepared = CONNECTOME / 'prepared'
    summary = json.loads((prepared / 'summary.json').read_text())
    graph = sparse.load_npz(prepared / 'adjacency.npz')
    transition = sparse.load_npz(prepared / 'transition.npz')
    nodes = pd.read_csv(prepared / 'nodes.csv.gz', dtype={'root_id': 'string'})
    layout = np.load(prepared / 'layout.npz', allow_pickle=False)
    assert len(nodes) == summary['neurons'] == 139255
    assert graph.shape == transition.shape == (len(nodes), len(nodes))
    assert graph.nnz == summary['neuron_pairs']
    assert graph.data.sum(dtype=np.float64) == summary['synapses']
    assert len(layout['points']) == summary['coordinate_points']
    assert len(np.unique(layout['point_nodes'])) == len(nodes)
    assert np.isfinite(layout['points']).all()
    edge_rows, edge_cols = layout['edges'].T
    assert np.asarray(graph[edge_rows, edge_cols]).min() > 0
    row_sums = np.asarray(transition.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums[row_sums > 0], 1, atol=2e-6)

    pulse = ConnectomePulse(transition)
    stimulation = np.flatnonzero((nodes['class'] == 'ALPN').to_numpy())
    pulse.reset(stimulation)
    pulse_rows = [{'iteration': 0, 'mass': float(pulse.activity.sum()),
                   'active_neurons': len(stimulation), 'compute_ms': 0.0}]
    initial_first = None
    for iteration in range(1, 31):
        start = perf_counter()
        activity = pulse.step()
        elapsed = (perf_counter() - start) * 1000
        assert np.isfinite(activity).all() and np.all(activity >= 0)
        assert activity.sum() <= pulse_rows[-1]['mass'] + .001
        if iteration == 1:
            initial_first = activity.copy()
        pulse_rows.append({'iteration': iteration, 'mass': float(activity.sum()),
                           'active_neurons': int(np.count_nonzero(activity > 1e-8)), 'compute_ms': elapsed})
    pulse.reset(stimulation)
    np.testing.assert_array_equal(pulse.step(), initial_first)
    pulse_frame = pd.DataFrame(pulse_rows)
    pulse_frame.to_csv(output / 'connectome_pulse.csv', index=False)

    lab = BrainLab()
    catalog = lab.alert_catalog()
    known = [r['index'] for r in catalog if not r['unseen']][:25]
    unseen = [r['index'] for r in catalog if r['unseen']][:25]
    trace_rows = []
    for index in known + unseen:
        trace = lab.trace_alert(index)
        expected = lab.pipeline.encode(lab.test.iloc[[index]])
        np.testing.assert_array_equal(trace['winner_indices'], expected.indices)
        assert len(trace['kc_activations']) == lab.pipeline.encoder.kc_dim
        assert trace['active_kcs'] == lab.pipeline.encoder.top_k
        trace_rows.append({'index': index, 'alert_id': trace['alert']['alert_id'],
                           'unseen': index in unseen, 'active_kcs': trace['active_kcs'],
                           'novelty': trace['novelty']['novelty_score'],
                           'is_novel': trace['novelty']['is_novel'], **trace['timing_ms']})
    trace_frame = pd.DataFrame(trace_rows)
    trace_frame.to_csv(output / 'fly_traces.csv', index=False)
    write_json(output / 'example_known_trace.json', lab.trace_alert(known[0]))
    write_json(output / 'example_unseen_trace.json', lab.trace_alert(unseen[0]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(pulse_frame.iteration, pulse_frame.mass, marker='.', color='#147d92')
    axes[0].set(xlabel='Abstract diffusion iteration', ylabel='Total signal mass (arbitrary units)',
                title='Real FlyWire graph · ALPN impulse')
    axes[1].plot(pulse_frame.iteration, pulse_frame.active_neurons, color='#8956a8')
    axes[1].set(xlabel='Abstract diffusion iteration', ylabel='Neurons with amplitude > 1e-8',
                title='Full graph computation')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / 'connectome_pulse.png', dpi=170)
    plt.close(fig)
    report = {'passed': True, 'environment': environment_manifest(), 'model_run': lab.run_path.name,
              'connectome': summary, 'source_manifest': manifest,
              'validation': {'raw_checksums': 'passed', 'graph_integrity': 'passed',
                             'rendered_edges_are_real': True, 'mass_decay_30_steps': True,
                             'pulse_repeat_deterministic': True, 'live_traces_equal_core': len(trace_rows)},
              'pulse': {'initial_ALPN_neurons': len(stimulation), 'first_step_active': pulse_rows[1]['active_neurons'],
                        'median_compute_ms': pulse_frame.compute_ms.iloc[1:].median(),
                        'p95_compute_ms': pulse_frame.compute_ms.iloc[1:].quantile(.95),
                        'transition_storage_bytes': transition.data.nbytes + transition.indices.nbytes + transition.indptr.nbytes},
              'fly_trace': {'median_total_ms': trace_frame.total.median(), 'p95_total_ms': trace_frame.total.quantile(.95),
                            'active_kcs_min': trace_frame.active_kcs.min(), 'active_kcs_max': trace_frame.active_kcs.max(),
                            'known_mean_novelty': trace_frame.loc[~trace_frame.unseen, 'novelty'].mean(),
                            'unseen_mean_novelty': trace_frame.loc[trace_frame.unseen, 'novelty'].mean()},
              'interpretation': 'Graph diffusion is abstract and unsigned. Trace timings include observability overhead. '
                                'The 50-alert trace subset is a consistency check, not a new generalization experiment.'}
    write_json(output / 'validation.json', report)
    write_json(ROOT / 'results/latest_brain_validation.json', {'path': output.name})
    print(json.dumps({'output': str(output), 'checks': report['validation'],
                      'pulse': report['pulse'], 'fly_trace': report['fly_trace']}, default=float, indent=2))


if __name__ == '__main__':
    main()
