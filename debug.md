# Debug

```
PYTHONFAULTHANDLER=1 ./derived.noindex/dist/MavisCorrector.plugin/Contents/MacOS/MavisCorrector --http-service --model-path $(pwd)/t5-small-spoken-typo.ct2

pkill -ABRT -f MavisCorrector
```

# Bugs

When using the development build, the process hangs waiting for data to come back from a multiprocess child for the spell checker.

However, this does not happen when using the release build that copies all files in place.

The grandchild process (mavis->multiproc->grandchild) seems defunct, but nothing interesting is returned the stack trace.

```
Current thread 0x0000000205d30840 (most recent call first):
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/connection.py", line 395 in _recv
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/connection.py", line 430 in _recv_bytes
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/connection.py", line 216 in recv_bytes
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/queues.py", line 103 in get
  File "./mavis-corrector/venv/lib/python3.12/site-packages/stc/spell_checker.py", line 653 in _get
  File "./mavis-corrector/mavis/app/mavis_corrector.py", line 234 in correct
  File "./mavis-corrector/mavis/app/mavis_corrector.py", line 94 in do_GET
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/server.py", line 424 in handle_one_request
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/server.py", line 436 in handle
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/socketserver.py", line 766 in __init__
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/socketserver.py", line 362 in finish_request
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/socketserver.py", line 349 in process_request
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/socketserver.py", line 318 in _handle_request_noblock
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/socketserver.py", line 240 in serve_forever
  File "./mavis-corrector/mavis/app/mavis_corrector.py", line 70 in serve_forever
  File "./mavis-corrector/mavis/app/mavis_corrector.py", line 124 in run_http
  File "./mavis-corrector/mavis/app/mavis_corrector.py", line 303 in http_service
  File "./mavis-corrector/mavis/app/mavis_corrector.py", line 319 in main
  File "./mavis-corrector/MavisCorrector.py", line 10 in <module>
  File "./mavis-corrector/derived.noindex/dist/MavisCorrector.plugin/Contents/Resources/__boot__.py", line 143 in _run
  File "./mavis-corrector/derived.noindex/dist/MavisCorrector.plugin/Contents/Resources/__boot__.py", line 149 in <module>
```