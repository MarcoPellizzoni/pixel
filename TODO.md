# TODO

1. Fix exception quando cambio valore alla gamma :

```
exception calling callback for <Future at 0x72455cba46e0 state=finished raised ValueError>
Traceback (most recent call last):
  File "/home/marco/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/concurrent/futures/_base.py", line 340, in _invoke_callbacks
    callback(self)
  File "/home/marco/pixel/.venv/lib/python3.12/site-packages/flet/controls/page.py", line 870, in _on_completion
    raise exception
  File "/home/marco/pixel/src/pixel/ui/app.py", line 597, in _change_pipeline
    self._refresh()
  File "/home/marco/pixel/src/pixel/ui/app.py", line 664, in _refresh
    self._pipeline.show_steps(session.applied, session.pipeline_text)
                              ^^^^^^^^^^^^^^^
  File "/home/marco/pixel/src/pixel/ui/session.py", line 99, in applied
    return tuple(
           ^^^^^^
  File "/home/marco/pixel/src/pixel/ui/session.py", line 101, in <genexpr>
    for invocation, result in zip(self._steps, self._results, strict=True)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1
exception calling callback for <Future at 0x72455cccf380 state=finished raised ValueError>
Traceback (most recent call last):
  File "/home/marco/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/concurrent/futures/_base.py", line 340, in _invoke_callbacks
    callback(self)
  File "/home/marco/pixel/.venv/lib/python3.12/site-packages/flet/controls/page.py", line 870, in _on_completion
    raise exception
  File "/home/marco/pixel/src/pixel/ui/app.py", line 597, in _change_pipeline
    self._refresh()
  File "/home/marco/pixel/src/pixel/ui/app.py", line 664, in _refresh
    self._pipeline.show_steps(session.applied, session.pipeline_text)
                              ^^^^^^^^^^^^^^^
  File "/home/marco/pixel/src/pixel/ui/session.py", line 99, in applied
    return tuple(
           ^^^^^^
  File "/home/marco/pixel/src/pixel/ui/session.py", line 101, in <genexpr>
    for invocation, result in zip(self._steps, self._results, strict=True)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zip() argument 2 is longer than argument 1
```

2. Fare un passata di tutto il codice

3. Fare una wiki

4. creare un apk android e un eseguibile per windows

