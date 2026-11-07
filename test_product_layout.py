import pandas as pd
import plotly.graph_objects as go
import pytest
import agent_core as ac
import geomag_plotter as plots
from chat_session import Request


def test_component_overlay_is_not_a_station_overlay():
    contract = Request('Наложи X и Y из одной таблицы на общей шкале.')
    assert contract.check_call('plot_overlay', {}, ac.FrameStore())
    assert contract.check_call('plot_components', {'layout': 'overlay'}, ac.FrameStore()) is None
    schemas = ac.select_tools_for_query(contract.query)
    assert 'plot_components' in {s['function']['name'] for s in schemas}


def test_layout_reaches_plotter(monkeypatch):
    store = ac.FrameStore()
    store.put('raw:bou:2024-09-10', pd.DataFrame({'X': [1.], 'Y': [2.]}))
    seen = {}
    def plot(frame, **kwargs):
        seen.update(kwargs)
        return {'error': 'test_stop'}
    monkeypatch.setattr(ac.plotter, 'plot_components', plot)
    ac._handle_plot_components({'df_descriptor': 'raw:bou:2024-09-10', 'components': ['X', 'Y'], 'layout': 'stacked'}, store)
    assert seen['layout'] == 'stacked'


@pytest.mark.parametrize('station', ['BOU', 'IRT', 'ABG'])
def test_explicit_ground_source_keeps_loader_with_statistics(station):
    schemas = ac.select_tools_for_query(f'Загрузи INTERMAGNET {station} за 2024-09-10 и получи статистику H инструментом.')
    assert {'fetch_observatory_data', 'get_statistics'} <= {s['function']['name'] for s in schemas}


@pytest.mark.parametrize('layout,axes', [('stacked', ['y', 'y2', 'y3']), ('overlay', [None]*3), ('auto', [None]*3)])
def test_explicit_layout_controls_real_figure(layout, axes, tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(go.Figure, 'write_html', lambda self, *a, **k: figures.append(self))
    frame = pd.DataFrame({'timestamp': pd.date_range('2024-09-10', periods=3, freq='min'),
                          'X': [1, 2, 3], 'Y': [10, 11, 12], 'Z': [100, 101, 102]})
    result = plots.plot_components(frame, ['X', 'Y', 'Z'], output_dir=tmp_path, layout=layout)
    assert not isinstance(result, dict), result
    assert [t.yaxis for t in figures[-1].data] == axes
    assert list(figures[-1].data[1].y) == [10, 11, 12]
    if layout == 'stacked':
        fig = figures[-1]
        assert fig.layout.yaxis.domain[0] > fig.layout.yaxis2.domain[1]
        assert fig.layout.yaxis2.domain[0] > fig.layout.yaxis3.domain[1]
