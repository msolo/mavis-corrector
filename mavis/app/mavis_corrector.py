#!/usr/bin/env python3

import builtins
import sys

__print = builtins.print


def _print(*args, **kargs):
    kargs["file"] = sys.stderr
    __print(*args, **kargs)


builtins.print = _print

import os

import argparse
import functools
import json
import logging
import subprocess
import threading
import time

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from mavis.stc import spell_checker

from mavis.app import service_discovery
from mavis.corrector.inference import (
    InferencerCt2,
    top5,
    fix_terminal_punctuation,
)

from mavis.corrector.text import (
    norm_text_eval,
)


test_fragments = [
    "this simple exampel shoudl make things comoile at least",
    "its actually kidn of hard to just type crap",
    "maybe thats good manbe im better tha this than i thought",
    "doe questions seem to work?",
]


class MavisHttpServer(HTTPServer):
    zc = None

    def server_bind(self):
        super().server_bind()
        # Handle binding to a random port.
        self.server_port = self.socket.getsockname()[1]

    def server_activate(self):
        super().server_activate()
        self.zc = service_discovery.MavisZeroConf(
            "_mavis-corrector",
            instance_name=service_discovery.get_computer_name(),
            port=self.server_port,
        )
        self.zc.advertise()

    def serve_forever(self, poll_interval=0.5):
        try:
            # Override serve_forever, otherwise we miss Control-C
            return super().serve_forever(poll_interval)
        finally:
            self.zc.close()

    def handle_error(self, request, client_address):
        if type(sys.exception()) in (ConnectionResetError,):
            return
        return super().handle_error(request, client_address)


# Custom request handler class
class RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    fmap = {}

    # Handle GET requests
    def do_GET(self):
        reply = {"error": None, "return": None, "id": int(time.time() * 1e9)}
        try:
            uri = urlparse(self.path)
            query_params = parse_qs(uri.query)
            # Flatten the query params, as parse_qs returns lists
            kwargs = {key: value[0] for key, value in query_params.items()}
            method_name = os.path.basename(uri.path)
            f = self.__class__.fmap[method_name]
            reply["return"] = f(**kwargs)
        except Exception as e:
            reply["error"] = str(e)
            logging.exception(e)

        try:
            data = json.dumps(reply).encode("utf-8") + b"\n"
        except Exception:
            # Something awful happened.
            data = '{"error": "Serialization failed", "return":null}\n'.encode("utf-8")

        # Prepare the response headers
        if reply["error"]:
            self.send_response(500)
        else:
            self.send_response(200)

        self.send_header("Content-type", "application/json")
        self.send_header("Content-length", len(data))  # For persistent connection
        self.end_headers()
        # Send the JSON response
        self.wfile.write(data)
        self.wfile.flush()

    def log_message(self, format, *args):
        if CLI_ARGS.http_logging_enabled:
            return super().log_message(format, *args)


# Server initialization
def run_http(server_class=MavisHttpServer, handler_class=RequestHandler, port=0):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting HTTP server on port {httpd.server_port}...")
    # NOTE: We run the http server in a thread so the main thread can correctly # handle stdin and signals from the parent. This enables proper cleanup
    # just as we have in the json-over-stdio mode.
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        for line in sys.stdin:
            pass
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    # Make sure to wait for the server to tear down and unregister.
    t.join()
    sys.exit(0)


# Hack to round this to
def norm_timestamp_to_ms(t):
    return int(t * 1000) / 1000


CLI_ARGS = None


def main():
    global CLI_ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", action="store_true")
    ap.add_argument("--json-service", action="store_true")
    ap.add_argument("--http-service", action="store_true")
    ap.add_argument("--http-port", default=0, type=int)
    ap.add_argument(
        "--http-logging-enabled",
        action="store_true",
        help="Enable http logging for debug purposes.",
    )
    ap.add_argument(
        "--app-logging-enabled",
        action="store_true",
        help="Enable application logging for debug purposes.",
    )
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--test-profile", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--allow-online", action="store_true")
    ap.add_argument("--model-path", help="A local directory containing a Ct2 model")
    ap.add_argument(
        "--extra-proper-nouns", help="A file containing additional proper nouns."
    )
    ap.add_argument("--eval-input", default=None)
    ap.add_argument("--eval-output", default=None)

    args = ap.parse_args()
    CLI_ARGS = args
    if not args.allow_online:
        os.environ.update(
            {
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            }
        )
    # We don't actually use tokenizer paralleism, but this squelches a loud, spurious warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if not args.model_path:
        model_name = "t5-small-spoken-typo.ct2"
        args.model_path = os.path.join(os.environ["RESOURCEPATH"], "models", model_name)

    fac = InferencerCt2(
        model_path=args.model_path,
    )

    correct_best = functools.partial(fac.correct, num_return_sequences=32)
    correct_profile = correct_best

    def test():
        for t in test_fragments + [". ".join(test_fragments)]:
            correct(t)

    def test_model():
        for t in test_fragments + [". ".join(test_fragments)]:
            correct_best(t)

    def test_profile():
        t0 = time.time()
        for t in test_fragments + [". ".join(test_fragments)]:
            correct_profile(t)
        tE = time.time() - t0
        print(f"wall time: {tE:05.3f}")

    # Take a JSONL log file from iMavis and collect the completions items.
    def eval_log_items(input_file, output_file):
        from mavis.corrector import common

        if not output_file:
            fdir, fname = os.path.split(input_file)
            fname, fext = os.path.splitext(fname)
            output_file = os.path.join(fdir, fname + "-eval" + fext)

        for item in common.read_jsonl(input_file):
            text = item["text"]
            corrections = correct(text)
            ci = {"text": text, "corrections": corrections}
            common.append_jsonl(output_file, [ci])

    def profile():
        import cProfile

        cProfile.runctx(
            "test_profile()", globals(), {"test_profile": test_profile}, "py.prof"
        )

    def correct_ml_only(text):
        t0 = time.time()
        results = correct_best(text)
        tElapsed = time.time() - t0
        _top5 = top5(results)
        print(f"best ({tElapsed:5.3f})", _top5)
        return _top5

    if args.extra_proper_nouns:
        try:
            names = [
                x.strip()
                for x in open(args.extra_proper_nouns).read().splitlines()
                if x.strip()
            ]
            spell_checker.update_proper_nouns(names)
        except FileNotFoundError as e:
            print("Missing file:", args.extra_proper_nouns)
        except Exception as e:
            print("Unhandled exception:", e)

    qcache = spell_checker.BoundedLRU(100)
    speller = spell_checker.SpellChecker()
    speller.start()

    def correct(text):
        log_entry = {"text": text}

        text_norm = norm_text_eval(text)
        results = qcache.get(text_norm)
        if results:
            return results

        # process in background - it's also CPU-bound
        wait_spelling_result = speller.correct_spelling_async(text)

        t0 = time.time()
        results = correct_best(text)

        spell_result = wait_spelling_result()
        # FIXME: this API is fugly
        respell_norm = fix_terminal_punctuation([norm_text_eval(spell_result)], text)[0]
        if respell_norm not in results:
            # Based on experiments, this is normally middle of the pack.
            results[2:2] = [respell_norm]
            # In case we have too many, trim it back - we lose about ~1%.
            results = results[:5]

        tElapsed = time.time() - t0
        log_entry["ml-results"] = results
        log_entry["ml-time"] = norm_timestamp_to_ms(tElapsed)

        log_entry["spell-results"] = [respell_norm]
        log_entry["spell-time"] = norm_timestamp_to_ms(0)

        log_entry["total-time"] = sum(
            [v for k, v in log_entry.items() if k.endswith("-time")]
        )
        if args.app_logging_enabled:
            print(json.dumps(log_entry, sort_keys=True), file=sys.stderr)
        qcache[text_norm] = results
        return results

    def ping():
        return "pong"

    def test_error():
        raise Exception("test error")

    fmap = {
        "test_error": test_error,
        "correct": correct,
        "correct_ml_only": correct_ml_only,
        "correct_spell_only": spell_checker.correct_text,
        "ping": ping,
    }

    def json_service():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            resp = {"error": "invalid message", "return": None}
            try:
                req = json.loads(line)
            except Exception as e:
                req = None
                # This could leak I suppose.
                resp = {"error": str(e), "return": None}

            if not req:
                sys.stdout.write(json.dumps(resp))
                sys.stdout.write("\n")
                sys.stdout.flush()
                continue

            f = fmap.get(req.get("method"))
            kwargs = req.get("args", {})
            if not f:
                resp = {"error": "no such method"}
            else:
                try:
                    r = f(**kwargs)
                    resp = {"error": None, "return": r}
                except Exception as e:
                    # This could leak I suppose.
                    resp = {"error": str(e), "return": None}
            sys.stdout.write(json.dumps(resp))
            sys.stdout.write("\n")
            sys.stdout.flush()

    def http_service(port):
        RequestHandler.fmap = fmap
        run_http(handler_class=RequestHandler, port=port)

    def service():
        for line in sys.stdin:
            if line:
                correct(line.strip())

    # Force model init and warmup by running some evals.
    test_model()

    try:
        if args.service:
            service()
        elif args.json_service:
            json_service()
        elif args.http_service:
            # As long as we are on AC power, don't sleep this process.
            caffeinate = subprocess.Popen(
                args=["/usr/bin/caffeinate", "-s", "-w", str(os.getpid())]
            )
            http_service(args.http_port)
        elif args.test:
            test()
        elif args.test_profile:
            test_profile()
        elif args.profile:
            profile()
        elif args.eval_input:
            eval_log_items(args.eval_input, args.eval_output)

    except KeyboardInterrupt:
        pass
    finally:
        speller.shutdown()


if __name__ == "__main__":
    main()
