"""Models load from the local cache first and only download when missing."""
from noor.model_loading import load_local_first


def test_uses_local_cache_without_network():
    calls = []

    def factory(name, **kwargs):
        calls.append(kwargs)
        return f"model:{name}"

    assert load_local_first(factory, "m") == "model:m"
    assert calls == [{"local_files_only": True}]  # one call, cache only


def test_downloads_when_not_cached():
    calls = []

    def factory(name, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in cache")
        return f"downloaded:{name}"

    assert load_local_first(factory, "m") == "downloaded:m"
    assert calls == [{"local_files_only": True}, {}]


def test_concurrent_requests_load_the_model_once():
    """Reproduces the race seen in the app: warm-up and a question at once."""
    import threading
    import time

    from noor.model_loading import get_model

    loads = []

    def slow_factory(name, **kwargs):
        loads.append(kwargs)
        time.sleep(0.2)  # a load that's still in progress when others arrive
        return object()

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(
            get_model("test-race", slow_factory, "m", "cpu")))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(loads) == 1                     # loaded exactly once
    assert len({id(r) for r in results}) == 1  # everyone shares that instance
    assert loads[0] == {"local_files_only": True, "device": "cpu"}
