# pixel

Composable, step-based image processing, described with the pipe:

```bash
uv run pixel run photo.jpg "remove-background | grayscale | pen-sketch"
```

Works on any image. 22 steps available, from the basics (resize, crop, rotate)
to the stylisations (pen drawing, pencil, cartoon), composable in any order.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv sync
```

Only the `remove-background` step downloads a segmentation model (~180 MB) on
first use; it is then cached in `~/.u2net/`. Every other step works right away.

## Usage

```bash
# The default pipeline: isolate the subject, greyscale, pen drawing
uv run pixel run g1.jpeg

# A hand-written pipeline
uv run pixel run photo.jpg "resize:width=1200 | auto-contrast | cartoon"

# List the available steps
uv run pixel steps

# A step's parameters, with their default values
uv run pixel describe pen-sketch
```

### The syntax

```
step1:parameter=value,other=value | step2 | step3:parameter=value
```

- steps are separated by `|`;
- a step's parameters open with `:` and are separated by `,`;
- names are written hyphenated (`ink-threshold`), colours in hexadecimal
  (`#ffffff`), booleans as `true`/`false`;
- whitespace around the separators is free.

Whatever you leave out keeps its default value, and what `describe` shows is
exactly what `run` accepts.

### Shell pipes

With `-` as the input image and `--stdout` as the output, several invocations
chain together like any other Unix command:

```bash
cat photo.jpg \
  | uv run pixel run - "resize:width=800" --stdout \
  | uv run pixel run - "sepia | vignette" --stdout \
  > result.png
```

Progress messages go to standard error, so they do not pollute the image stream.

### Seeing the intermediate stages

```bash
uv run pixel run photo.jpg "denoise | edges" --save-steps
```

This also writes `01_denoise.png` and `02_edges.png` into `output/`, which helps
work out which stage needs adjusting.

## The graphical editor

```bash
uv run pixel-editor                 # open the editor
uv run pixel-editor photo.jpg       # open it on a given image
uv run pixel-editor --web           # serve it to a browser instead
uv run pixel-editor --fresh         # ignore the work from last time
```

The window has the photo in the middle, the step catalogue on the left and the
pipeline on the right. Anything the command line can express can be built here,
and the pipeline panel shows the result written in exactly the syntax above,
ready to be copied into a terminal.

**Adding a step.** Drag it onto the pipeline, or press its **+**. Either way the
photo updates straight away.

**Changing a step.** Press **⚙** on a step in the pipeline to open its settings.
Every parameter gets the control that suits it — a switch, a menu, a box — with
its explanation underneath. Typed values apply when you leave the field or press
Enter; switches and menus apply at once.

**Reordering.** The arrows on each step move it up or down. Order matters:
greying then inverting is not the same as inverting then greying, so the photo
changes when you move something.

**Removing.** The **×** deletes any step, not only the last one. Everything after
it is recomputed.

**Undo and Redo** step back and forward through every kind of change — an added
step, a move, a removal, a retyped parameter, even a reset. Making a fresh change
after undoing discards the branch you left. **Reset** empties the pipeline, and
**Save** writes the result at full resolution with transparency intact, not the
scaled-down copy on screen.

**Arranging the window.** Drag the divider beside either side panel to widen or
narrow it, within limits that keep it usable and keep the photo visible. The two
buttons at the far ends of the toolbar put each panel away and bring it back.

**Saving a session.** The folder icon beside Save holds **Save session as…** and
**Open session…**. A session is a file you name and keep: it holds the photo's
full path and the pipeline — every step, its parameters and their order — in the
same syntax shown in the panel. Opening one loads that photo again and replays
the steps over it, so you get back exactly what you had.

Because no pixels are stored, a session doubles as a recipe: open one while a
different photo is loaded and its photo has since moved, and the steps are
applied to what you have open instead. The files are a couple of hundred bytes,
readable, and worth keeping in version control.

**Picking up where you left off.** The editor writes down which photo was open,
the pipeline applied to it and how the window was arranged, and restores all of
it next time. What is stored is a path, the pipeline in the syntax above and a
few numbers — no images — so the file stays small enough to read by hand, and
reopening simply replays the pipeline over the original. Start with `--fresh` to
skip it. It lives in `~/.config/pixel/workspace.json`.

**Help.** The **?** in the toolbar lists every step; the **?** beside a step, or
the link inside its settings, explains that one and each of its parameters. All
of it is generated from the same catalogue as `pixel steps` and `pixel describe`,
so the two interfaces cannot disagree.

Transparency is drawn over a grey chequerboard, so a `remove-background` cut-out
is visible for what it is.

> On Linux the desktop window needs the system's graphical libraries, which a
> plain WSL or headless install does not have (the symptom is a missing
> `libsecret-1.so.0`). Either install them —
> `sudo apt install libsecret-1-0 libgtk-3-0 libmpv2` — or use `--web`, which
> needs none of them.

## The steps

| Family | Steps |
| --- | --- |
| Geometry | `resize` `crop` `rotate` `flip` |
| Colour | `grayscale` `sepia` `invert` `saturation` `posterize` |
| Tone | `brightness-contrast` `gamma` `auto-contrast` `threshold` |
| Filters | `blur` `sharpen` `denoise` `edges` |
| Segmentation | `remove-background` |
| Artistic | `pen-sketch` `pencil-sketch` `cartoon` `vignette` |

A few useful combinations:

```bash
# A cleaned-up, printable portrait
"remove-background | auto-contrast | sharpen:amount=0.8"

# A photographed document, turned into a scan
"denoise | threshold:method=adaptive,block-size=41"

# A sparser, more nervous pen drawing
"remove-background | pen-sketch:ink-threshold=0.7,dog-sigma=0.9"

# A square thumbnail
"resize:width=400,height=400,fit=cover | crop:width=400,height=400"
```

## Use as a library

Composing the steps directly:

```python
from pathlib import Path

from pixel import (
    GrayscaleConfig, GrayscaleStep, ImagePipeline,
    PenSketchConfig, PenSketchStep, load_image, save_image,
)

image = load_image(Path("photo.jpg"))
pipeline = ImagePipeline([
    GrayscaleStep(GrayscaleConfig()),
    PenSketchStep(PenSketchConfig(ink_threshold=0.7)),
])
save_image(pipeline.run(image).final_image, Path("drawing.png"))
```

Or from the same string used on the command line, so the pipeline can live in a
configuration file:

```python
from pixel import build_pipeline

pipeline = build_pipeline("resize:width=800 | grayscale | pen-sketch")
```

## How the code is organised

An `src/` layout, with one responsibility per module:

```
src/pixel/
├── domain.py        The `RGBAImage` type that flows through the whole pipeline
├── image_io.py      The only module touching the filesystem (files and streams)
├── pipeline.py      Runs a sequence of steps; does not know which ones exist
├── dsl.py           Reads the string "a:x=1 | b" and derives the steps from it
├── params.py        Converts textual parameters into typed configurations
├── registry.py      The catalogue: public name → step + configuration
├── errors.py        The errors a user can cause by writing a bad pipeline
├── cli.py           The run / steps / describe commands
├── steps/
    ├── base.py          The `ProcessingStep` protocol shared by all of them
    ├── geometry.py      shape and size
    ├── color.py         colour interpretation
    ├── tone.py          tone curve
    ├── filters.py       spatial filters
    ├── background.py    subject isolation
    ├── mask_cleanup.py  refinements to the cut-out mask
│   ├── artistic.py      stylisations
│   └── pen_sketch.py    pen drawing (the most elaborate effect)
└── ui/              The graphical editor
    ├── session.py       the pipeline, the picture it makes, undo and redo
    ├── preview.py       encoding an image for the screen
    ├── layout.py        panel widths and visibility, with their limits
    ├── workspace.py     remembering the work between one run and the next
    ├── sessions.py      named session files the user saves and opens
    ├── theme.py         colours, sizes and spacing
    ├── updates.py       redrawing a control only when it is on screen
    ├── app.py           wires the panels to the session
    ├── launcher.py      reads the command line and starts Flet
    └── components/      toolbar, library, canvas, pipeline, splitter,
                         parameters (the fields), help (the explanations)
```

The dependencies run one way only: `cli` → `registry` → `steps` → `domain`, and
`ui/app` → `ui/components` → `ui/theme` alongside `ui/app` → `ui/session` → the
processing library. The steps know neither the filesystem nor each other, so they
are testable with in-memory arrays and freely recombinable.

The editor adds no image processing of its own, and describes no step by hand.
Effects, parameters, defaults and explanations all come from `registry` and
`params`, derived by introspection from the steps' own configuration classes.
That is why a new step — or a new parameter on an existing one — turns up in the
library, in its settings panel, in the help and in `pixel describe` at once,
without either interface being touched.

A parameter's explanation lives beside its type, as an `Annotated` note:

```python
radius: Annotated[float, "How far the blur reaches, in pixels."] = 3.0
```

Putting it there rather than in a comment is what lets the window and the
terminal say the same thing about it.

### Adding a step

Two things are needed, and nothing else changes:

1. in the module of the right family, a configuration dataclass and a class with
   `name` and `apply(image) -> image`;
2. one line in `STEP_DEFINITIONS` inside [registry.py](src/pixel/registry.py).

The command line, the built-in help and the parameter conversion adapt by
themselves: names, types and default values are derived from the dataclass by
introspection.

## Packages used, and why

| Package | Role |
| --- | --- |
| **rembg** (U²-Net / IS-Net) | Subject segmentation. On an arbitrary photo the background can have exactly the same colours and textures as the subject: no threshold-based technique would hold up. |
| **OpenCV** | The bulk of the algorithms: filters, geometric transforms, CLAHE, thresholds, connected components. |
| **Pillow** | Reading/writing files and handling EXIF orientation. |
| **NumPy** | Representation and vectorised computation over pixel matrices. |
| **Typer** | Command-line interface. |
| **Flet** | The graphical editor's window, built on Flutter. `flet[desktop]` supplies the desktop client, `flet[web]` the browser one. |

Development tools: **pytest**, **Ruff** (linting and imports), **Pyright** (type
checking, the same engine as Pylance).

## Development

```bash
uv run pytest        # 496 tests
uv run ruff check .  # linting and import sorting
uv run pyright       # static type checking
```

### Type checking

**Pylance**, the VS Code extension, and **Pyright**, the command above, are the
same thing: Pylance is the closed-source build of Pyright's analysis engine.
Pylance cannot be run from the command line, nor installed as a Python package,
so CI uses Pyright — which produces the same diagnostics.

The rules live in `[tool.pyright]` inside [pyproject.toml](pyproject.toml), in
`strict` mode, and `.vscode/settings.json` points Pylance at the same
environment: **editor and terminal cannot diverge.** The check is clean across
the whole project.

There are exactly two suppressions, both on the `rembg` imports in
[background.py](src/pixel/steps/background.py), which ships no type annotations.
`reportUnnecessaryTypeIgnoreComment` is enabled as an error, so a suppression
that has become unnecessary fails the check instead of lingering.

Node does not need installing: `nodejs-wheel-binaries` provides it as a
development dependency.
