"""Endpoint regression tests for legacy configservice Flask app."""

import configservice as cs


class DummyProcess:
    """Simple subprocess stand-in used by endpoint tests."""

    def __init__(self, pid):
        self.pid = pid
        self._running = True
        self.killed = False

    def poll(self):
        return None if self._running else 0

    def kill(self):
        self.killed = True
        self._running = False

    def terminate(self):
        self._running = False

    def wait(self, timeout=None):
        self._running = False
        return 0


def test_restart_sequence_stop_then_start(monkeypatch):
    """Stopping and restarting a project should work without NameError regressions."""
    try:
        cs.scheduler.shutdown(wait=False)
    except Exception:
        pass

    cs.app.config['TESTING'] = True
    cs.globvars['processes'] = {}

    popen_calls = []

    def fake_popen(*args, **kwargs):
        process = DummyProcess(pid=1000 + len(popen_calls))
        popen_calls.append(process)
        return process

    monkeypatch.setattr(cs, 'load_config', lambda: {'nebul-two': {'port': '5011'}})
    monkeypatch.setattr(cs, 'read_process_output', lambda _process: None)
    monkeypatch.setattr(cs.subprocess, 'Popen', fake_popen)

    client = cs.app.test_client()

    first_start = client.get('/start?project=nebul-two')
    assert first_start.status_code == 200
    assert first_start.get_json()['message'] == 'Project nebul-two started'

    stop = client.get('/stop?project=nebul-two')
    assert stop.status_code == 200
    assert stop.get_json()['message'] == 'Project nebul-two stopped'

    second_start = client.get('/start?project=nebul-two')
    assert second_start.status_code == 200
    assert second_start.get_json()['message'] == 'Project nebul-two started'

    assert len(popen_calls) == 2
    assert cs.globvars['processes']['nebul-two'].poll() is None


def test_start_returns_already_running_without_spawning_second_process(monkeypatch):
    """Starting an already-running project should not spawn a duplicate process."""
    try:
        cs.scheduler.shutdown(wait=False)
    except Exception:
        pass

    cs.app.config['TESTING'] = True
    cs.globvars['processes'] = {}

    popen_calls = []

    def fake_popen(*args, **kwargs):
        process = DummyProcess(pid=2000 + len(popen_calls))
        popen_calls.append(process)
        return process

    monkeypatch.setattr(cs, 'load_config', lambda: {'nebul-two': {'port': '5011'}})
    monkeypatch.setattr(cs, 'read_process_output', lambda _process: None)
    monkeypatch.setattr(cs.subprocess, 'Popen', fake_popen)

    client = cs.app.test_client()

    first_start = client.get('/start?project=nebul-two')
    assert first_start.status_code == 200
    assert first_start.get_json()['message'] == 'Project nebul-two started'

    second_start = client.get('/start?project=nebul-two')
    assert second_start.status_code == 200
    assert second_start.get_json()['message'] == 'Project nebul-two is already running'

    assert len(popen_calls) == 1
