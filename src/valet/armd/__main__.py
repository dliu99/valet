from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .app import open_fake, open_sim, open_so101
from .check import run_check
from .server import make_server
from .settings import load


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="valet-armd", description="The SO-101 arm server: the only process on the servo bus.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, about in (("serve", "serve the HTTP API"), ("check", "the pre-run check")):
        s = sub.add_parser(name, help=about)
        s.add_argument("--fake", action="store_true", help="a kinematic fake arm and box: no LeRobot, no serial port")
        s.add_argument("--config", type=Path, help="default: ./valet.toml when it exists")
        s.add_argument("--sim", action="store_true", help="MuJoCo physics and rendered cameras: no LeRobot, no serial port")
        s.add_argument("--allow-motion", action="store_true", help="with --fake or --sim only: let motions run without editing valet.toml")
    a = p.parse_args(argv)
    config = a.config or (Path("valet.toml") if Path("valet.toml").is_file() else None)
    settings = load(config)
    if a.allow_motion:
        if not (a.fake or a.sim):
            p.error("--allow-motion is for --fake and --sim; for the arm, set allow_motion in valet.toml")
        settings = replace(settings, allow_motion=True)

    def open_app():
        if a.sim:
            return open_sim(settings)
        return open_fake(settings) if a.fake else open_so101(settings)

    if a.cmd == "check":
        return 0 if run_check(open_app) else 1

    app = open_app()
    host, port = settings.listen
    server = make_server(app, host, port)
    print(f"armd ({app.kind}) on http://{host}:{port}, allow_motion={settings.allow_motion}, config {config or 'defaults'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
