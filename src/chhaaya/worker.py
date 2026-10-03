import logging
import signal
import threading

log = logging.getLogger("chhaaya.worker")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    log.info("worker started")
    stop.wait()
    log.info("worker stopped")


if __name__ == "__main__":
    main()
