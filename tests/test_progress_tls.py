"""TLS handling on the progress-reporting channel.

The progress endpoint is the gpustack server's own URL, so on an HTTPS deployment
it may present a certificate signed by a private CA. The runner container is a
separate image from gpustack's, so that CA is not necessarily in its trust store.

Three behaviors are pinned here:

* A failed progress update must NOT escape into guidellm's callback. It used to
  raise RuntimeError, which killed the whole benchmark over lost telemetry --
  exactly what an unverifiable server certificate triggered.
* ``ca_cert`` must reach the session's connector, and must stay scoped to it:
  the point of the option is that it does not disturb the trust store the rest
  of the process uses.
* ``insecure_skip_tls_verify`` must do the same, and neither must replace the
  default connector on a plain-HTTP URL where either would be meaningless.
"""

import asyncio
import ssl

import aiohttp
import pytest

from benchmark_runner.progress import ServerBenchmarkerProgress


@pytest.fixture
def ca_file(tmp_path):
    """A real, loadable CA file: create_default_context parses what it is given."""
    import subprocess

    cert = tmp_path / "ca.crt"
    key = tmp_path / "ca.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=gpustack.internal",
        ],
        check=True,
        capture_output=True,
    )
    return str(cert)


def _connector(url: str, *, insecure: bool = False, ca_cert: str = None):
    progress = ServerBenchmarkerProgress(
        progress_url=url, ca_cert=ca_cert, insecure_skip_tls_verify=insecure
    )

    async def build():
        await progress.on_initialize(None)
        session = progress.session
        # Read the connector's state before closing: closing the session closes
        # the connector it owns. `_ssl` is aiohttp-internal, but it is the only
        # way to observe what the connector was built with.
        state = (
            session.connector._ssl
            if isinstance(session.connector, aiohttp.TCPConnector)
            else None
        )
        await session.close()
        return state

    return asyncio.run(build())


def _connector_ssl_state(url: str, insecure: bool):
    return _connector(url, insecure=insecure)


def test_https_with_the_flag_disables_verification():
    assert _connector_ssl_state("https://gpustack.internal/state", True) is False


def test_https_without_the_flag_keeps_verification():
    # Default connector: verification on, against whatever the image trusts.
    assert _connector_ssl_state("https://gpustack.internal/state", False) is True


def test_a_ca_cert_builds_a_context_from_that_bundle(ca_file):
    state = _connector("https://gpustack.internal/state", ca_cert=ca_file)

    assert isinstance(state, ssl.SSLContext)
    assert state.verify_mode == ssl.CERT_REQUIRED
    assert state.check_hostname is True
    # Exactly the handed-over bundle -- not the system store, and not the system
    # store plus this.
    assert len(state.get_ca_certs()) == 1


def test_a_ca_cert_does_not_touch_the_process_trust_store(ca_file):
    """The whole point of the option over SSL_CERT_FILE.

    Everything else in the runner -- a Hugging Face processor fetch, say -- must
    still verify against what the image trusts.
    """
    before = len(ssl.create_default_context().get_ca_certs())

    _connector("https://gpustack.internal/state", ca_cert=ca_file)

    assert len(ssl.create_default_context().get_ca_certs()) == before


def test_plain_http_ignores_a_ca_cert(ca_file):
    assert _connector("http://gpustack.internal/state", ca_cert=ca_file) is True


def test_insecure_wins_over_a_ca_cert(ca_file):
    """Both set is a contradiction; the explicit "cannot be verified" wins."""
    state = _connector(
        "https://gpustack.internal/state", insecure=True, ca_cert=ca_file
    )

    assert state is False


def test_plain_http_ignores_the_flag():
    assert _connector_ssl_state("http://gpustack.internal/state", True) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"insecure_skip_tls_verify": True},
        {"ca_cert": None},  # filled in by the test from the fixture
    ],
    ids=["insecure", "ca_cert"],
)
def test_a_reopened_session_gets_a_live_connector(kwargs, ca_file):
    """Each session builds its own connector.

    The multi-run modes (the auto-tune ramp, manual stages) reuse one progress
    object across runs and close the session after each, so a connector cached on
    the instance would come back already closed on the next run.
    """
    if "ca_cert" in kwargs:
        kwargs = {"ca_cert": ca_file}
    progress = ServerBenchmarkerProgress(
        progress_url="https://gpustack.internal/state", **kwargs
    )

    async def run_twice():
        progress._ensure_session()
        first = progress.session
        await progress.on_finalize()
        assert first.closed

        progress._ensure_session()
        second = progress.session
        assert second is not first
        assert not second.closed
        assert not second.connector.closed
        await progress.on_finalize()

    asyncio.run(run_twice())


def test_a_failed_update_does_not_reach_the_caller():
    """The regression that killed benchmarks.

    Nothing listens on this port, so the PATCH fails. guidellm calls
    on_benchmark_start from its own callback, so an exception here aborted a run
    that had already been measuring for an hour.
    """
    progress = ServerBenchmarkerProgress(
        progress_url="https://127.0.0.1:1/state", progress_auth="token"
    )

    async def go():
        await progress.on_initialize(None)
        await progress.on_benchmark_start(None)
        await progress.on_finalize()

    asyncio.run(go())  # must not raise

    # The throttle clock advances even on failure, so an endpoint that stays
    # unreachable is retried at the interval rather than on every callback.
    assert progress._last_update_ts > 0
    # ...and the last acknowledged progress is untouched, since nothing landed.
    assert progress._last_progress == -1.0
