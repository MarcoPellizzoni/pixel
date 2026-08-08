"""The application: it wires the panels to the editing session.

Single responsibility: own the session, react to what the panels report, and put
the result back on screen. It is the only module that knows both halves, which is
why the panels can stay unaware of editing and the session unaware of Flet.

Every edit follows the same path:

    a panel reports an action
        -> the session is changed
        -> `_refresh` redraws canvas, pipeline and toolbar from the session

Because the screen is always redrawn from the session rather than patched in
place, the interface cannot drift out of step with the picture.

Steps are run on a worker thread. Background removal takes seconds, and doing
that on the interface's own thread would freeze the window solid.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import flet as ft

from pixel.dsl import StepInvocation, parse_pipeline
from pixel.errors import PipelineDefinitionError
from pixel.image_io import load_image, save_image
from pixel.registry import get_definition
from pixel.ui import preview, sessions, theme, workspace
from pixel.ui.components import help as help_dialogs
from pixel.ui.components.canvas import EditorCanvas
from pixel.ui.components.library import StepLibraryPanel
from pixel.ui.components.pipeline import PipelinePanel
from pixel.ui.components.splitter import PanelSplitter
from pixel.ui.components.toolbar import EditorToolbar
from pixel.ui.layout import PanelLayout
from pixel.ui.session import EditingSession

# Extensions offered when opening an image.
IMAGE_EXTENSIONS: list[str] = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]

# Default name suggested when saving. PNG keeps the transparency a cut-out leaves
# behind, which a JPEG would silently flatten.
DEFAULT_SAVE_NAME: str = "edited.png"

# How long a message stays on screen, in milliseconds.
MESSAGE_DURATION: int = 4000


class ImageEditorApp:
    """The editor window, and the state behind it."""

    def __init__(
        self,
        page: ft.Page,
        initial_image: Path | None = None,
        restore: bool = True,
    ) -> None:
        """Assemble the window and open whichever image applies.

        Args:
            page: the Flet page to build the interface on.
            initial_image: an image to open at start-up. When given it wins over
                anything remembered from last time.
            restore: whether to pick up where the last run left off, when no
                image was named.
        """
        self._page = page
        self._session: EditingSession | None = None
        self._current_path: Path | None = None

        # Held for the whole of any change to the session. Every edit runs on a
        # worker thread, and two of them at once would interleave inside the
        # session: one emptying the cached results while the other refills it
        # leaves more results than steps, and the next redraw fails.
        #
        # It is easy to start two by accident. Pressing Enter in a settings field
        # both submits it and takes the focus away, and a slow step leaves plenty
        # of time to press something else. Serialising them here means the
        # session is only ever touched by one thing at a time, whatever the
        # interface allows.
        self._work_lock = asyncio.Lock()

        # What the editor was doing last time. The layout is taken from it right
        # away; the photo and its pipeline are restored once the window exists.
        self._workspace = workspace.load() if restore else workspace.Workspace()
        self._layout = self._workspace.layout

        # The file picker is a service rather than a control: it opens the
        # operating system's own dialog, so it has nothing to draw.
        self._file_picker = ft.FilePicker()
        page.services.append(self._file_picker)

        self._toolbar = EditorToolbar(
            on_open=self.open_image,
            on_save=self.save_image,
            on_undo=self.undo,
            on_redo=self.redo,
            on_reset=self.reset,
            on_help=self.show_overview,
            on_toggle_library=self.toggle_library,
            on_toggle_pipeline=self.toggle_pipeline,
            on_save_session=self.save_session,
            on_open_session=self.open_session,
        )
        self._library = StepLibraryPanel(
            on_apply_now=self.apply_step,
            on_help=self.show_step_help,
        )
        self._canvas = EditorCanvas()
        self._pipeline = PipelinePanel(
            on_step_dropped=self.apply_step,
            on_move=self.move_step,
            on_remove=self.remove_step,
            on_parameters_changed=self.set_step_parameters,
            on_help=self.show_step_help,
        )

        self._library_splitter = PanelSplitter(
            on_drag=self.resize_library, on_drag_end=self._remember_workspace
        )
        self._pipeline_splitter = PanelSplitter(
            on_drag=self.resize_pipeline, on_drag_end=self._remember_workspace
        )

        self._configure_page()
        self._build_layout()
        self._apply_layout()

        # Deferred to the event loop: the window has to exist before an image can
        # be drawn into it.
        if initial_image is not None:
            page.run_task(self._open_image, initial_image)
        elif self._workspace.image_path is not None:
            page.run_task(self._restore_workspace)

    # ------------------------------------------------------------------
    # Assembling the window
    # ------------------------------------------------------------------

    def _configure_page(self) -> None:
        """Apply the window's own settings: title, theme, size."""
        self._page.title = "pixel — image editor"
        self._page.theme_mode = ft.ThemeMode.DARK
        self._page.bgcolor = theme.CANVAS_BACKGROUND
        self._page.padding = 0
        self._page.spacing = 0

        # A window whose size is left unset is handed to the window manager with
        # nothing to go on, and on Linux that can produce one the user cannot
        # drag the edges of. Stating the size and the limits explicitly is what
        # makes it behave like an ordinary application window.
        window = self._page.window
        window.width = theme.WINDOW_WIDTH
        window.height = theme.WINDOW_HEIGHT
        window.min_width = theme.WINDOW_MIN_WIDTH
        window.min_height = theme.WINDOW_MIN_HEIGHT
        window.resizable = True
        window.maximizable = True

    def _build_layout(self) -> None:
        """Place the toolbar, the two panels and the canvas in the window."""
        self._page.add(
            ft.Column(
                controls=[
                    self._toolbar.control,
                    ft.Row(
                        controls=[
                            self._library.control,
                            self._library_splitter.control,
                            # The canvas is the only part that expands, so the
                            # side panels keep their width as the window resizes.
                            self._canvas.control,
                            self._pipeline_splitter.control,
                            self._pipeline.control,
                        ],
                        spacing=0,
                        expand=True,
                    ),
                ],
                spacing=0,
                expand=True,
            )
        )

    def _apply_layout(self) -> None:
        """Push the current layout onto the panels and their dividers.

        A hidden panel takes its divider with it: a handle for resizing something
        that is not there would be nothing but a trap.
        """
        self._library.control.width = self._layout.library_width
        self._library.control.visible = self._layout.library_visible
        self._library_splitter.control.visible = self._layout.library_visible

        self._pipeline.set_width(self._layout.pipeline_width)
        self._pipeline.control.visible = self._layout.pipeline_visible
        self._pipeline_splitter.control.visible = self._layout.pipeline_visible

        # The page is redrawn directly: `refresh` guards controls that may not be
        # on a page yet, which is a question that cannot arise for the page itself.
        self._page.update()

    # ------------------------------------------------------------------
    # The editor's commands
    #
    # These are what the panels call, and they are equally what a test drives.
    # Each is safe to invoke at any moment, including before an image has been
    # opened, because every one of them is reachable from the interface at all
    # times.
    # ------------------------------------------------------------------

    @property
    def session(self) -> EditingSession | None:
        """The editing session, or None while no image is open."""
        return self._session

    @property
    def layout(self) -> PanelLayout:
        """How the window is arranged: panel widths and what is showing."""
        return self._layout

    def open_image(self) -> None:
        """Ask the user for an image and open it."""
        self._page.run_task(self._choose_and_open)

    def save_image(self) -> None:
        """Ask the user where to write the current image, and write it."""
        self._page.run_task(self._choose_and_save)

    def apply_step(self, step_name: str) -> None:
        """Apply one step to the current image.

        Args:
            step_name: the catalogue name of the step to run.
        """
        self._page.run_task(self._apply_step, step_name)

    def move_step(self, index: int, destination: int) -> None:
        """Move a step to another position in the pipeline.

        Args:
            index: the step's current position, counting from 0.
            destination: where it should end up.
        """
        self._page.run_task(
            self._change_pipeline, lambda session: session.move(index, destination)
        )

    def remove_step(self, index: int) -> None:
        """Delete one step, wherever it sits in the pipeline.

        Args:
            index: the position of the step to remove.
        """
        self._page.run_task(
            self._change_pipeline, lambda session: session.remove_at(index)
        )

    def set_step_parameters(self, index: int, parameters: dict[str, str]) -> None:
        """Reconfigure a step already in the pipeline.

        Args:
            index: the position of the step to change.
            parameters: its new parameters, by hyphenated name.
        """
        self._page.run_task(
            self._change_pipeline,
            lambda session: session.replace_at(
                index, StepInvocation(session.step_at(index).name, parameters)
            ),
        )

    def show_step_help(self, step_name: str) -> None:
        """Explain one step and every setting it takes.

        Args:
            step_name: the catalogue name of the step.
        """
        try:
            definition = get_definition(step_name)
        except PipelineDefinitionError as error:
            self._notify(str(error), is_error=True)
            return

        self._page.show_dialog(help_dialogs.build_step_help(definition))

    def show_overview(self) -> None:
        """Show the help covering every step in the catalogue."""
        self._page.show_dialog(help_dialogs.build_overview())

    def undo(self) -> None:
        """Take back the last change to the pipeline, whatever it was."""
        self._page.run_task(self._change_pipeline, lambda session: session.undo())

    def redo(self) -> None:
        """Put back a change that was taken back."""
        self._page.run_task(self._change_pipeline, lambda session: session.redo())

    def save_session(self) -> None:
        """Ask for a name and write the pipeline to a session file."""
        self._page.run_task(self._choose_and_save_session)

    def open_session(self) -> None:
        """Ask for a session file and reapply the steps it holds."""
        self._page.run_task(self._choose_and_open_session)

    def save_session_to(self, path: Path) -> None:
        """Write the pipeline to a session file, without asking where.

        Choosing the file and writing it are kept apart so that the writing can
        be driven directly — by a test, or by anything else that already knows
        where the file should go.

        Args:
            path: the file to write.
        """
        self._page.run_task(self._save_session_to, path)

    def open_session_from(self, path: Path) -> None:
        """Reapply the session held in a file, without asking which.

        Args:
            path: the session file to read.
        """
        self._page.run_task(self._open_session_from, path)

    def toggle_library(self) -> None:
        """Show the step library if it is hidden, hide it if it is showing."""
        self._layout.toggle_library()
        self._apply_layout()
        self._remember_workspace()

    def toggle_pipeline(self) -> None:
        """Show the pipeline panel if it is hidden, hide it if it is showing."""
        self._layout.toggle_pipeline()
        self._apply_layout()
        self._remember_workspace()

    def reset(self) -> None:
        """Discard every step and go back to the opened image."""
        if self._session is None:
            return

        # Through the same serialised path as every other change: emptying the
        # pipeline while a step is still running in a worker thread would leave
        # the session describing one thing and holding another.
        self._page.run_task(self._reset_now)

    async def _reset_now(self) -> None:
        """Empty the pipeline, waiting for any change in flight to finish."""
        await self._change_pipeline(lambda session: session.reset())
        self._notify("Back to the original image")

    # ------------------------------------------------------------------
    # Named sessions
    # ------------------------------------------------------------------

    async def _choose_and_save_session(self) -> None:
        """Show the save dialog and write the session where asked."""
        session = self._session
        if session is None:
            self._notify("Open an image before saving a session", is_error=True)
            return

        destination = await self._file_picker.save_file(
            dialog_title="Save session",
            file_name=sessions.suggested_name(self._current_path),
            allowed_extensions=[sessions.SESSION_EXTENSION],
        )
        if destination is None:
            # The dialog was cancelled.
            return

        await self._save_session_to(Path(destination))

    async def _save_session_to(self, path: Path) -> None:
        """Write the current pipeline to a session file.

        Args:
            path: the file to write.
        """
        session = self._session
        if session is None:
            self._notify("Open an image before saving a session", is_error=True)
            return

        try:
            sessions.save(
                sessions.SavedSession(
                    image_path=self._current_path,
                    pipeline=session.pipeline_text,
                ),
                path,
            )
        except OSError as error:
            self._notify(f"Could not save the session: {error}", is_error=True)
            return

        self._notify(f"Session saved to {path.name}")

    async def _choose_and_open_session(self) -> None:
        """Show the open dialog and reapply the session that was chosen."""
        files = await self._file_picker.pick_files(
            dialog_title="Open session",
            allowed_extensions=[sessions.SESSION_EXTENSION],
        )
        if not files:
            return

        chosen = files[0].path
        if chosen is None:
            self._notify("That file has no path on this platform", is_error=True)
            return

        await self._open_session_from(Path(chosen))

    async def _open_session_from(self, path: Path) -> None:
        """Read a session file and replay what it holds.

        Args:
            path: the session file to read.
        """
        try:
            saved = sessions.load(path)
        except (OSError, sessions.SessionFileError) as error:
            self._notify(f"Could not open the session: {error}", is_error=True)
            return

        await self._apply_saved_session(saved)

    async def _apply_saved_session(self, saved: sessions.SavedSession) -> None:
        """Load a session's photo and replay its steps over it.

        The photo stored in the session is the one the steps are applied to. When
        it can no longer be found, the steps are applied to whatever is currently
        open instead — a saved session doubles as a recipe, and refusing to use
        one because its original photo has moved would throw that away.

        Args:
            saved: the session that was read from the file.
        """
        path = saved.image_path

        if path is not None and path.is_file():
            await self._open_image(path)
        elif self._session is None:
            self._notify(
                "That session's image could not be found, and none is open",
                is_error=True,
            )
            return
        else:
            self._notify("That session's image was not found; using the open one")
            await self._change_pipeline(lambda session: session.reset())

        if self._session is None or not saved.pipeline:
            return

        try:
            invocations = parse_pipeline(saved.pipeline)
        except PipelineDefinitionError as error:
            self._notify(f"That session could not be replayed: {error}", is_error=True)
            return

        await self._change_pipeline(lambda session: _replay(session, invocations))

    # ------------------------------------------------------------------
    # Rearranging the window
    # ------------------------------------------------------------------

    def resize_library(self, delta: float) -> None:
        """Widen or narrow the step library.

        The width is not written down on every drag report — that would mean a
        file write per pixel of movement. It is stored when the drag ends.

        Args:
            delta: how far the pointer moved, in pixels.
        """
        self._layout.resize_library(delta)
        self._apply_layout()

    def resize_pipeline(self, delta: float) -> None:
        """Widen or narrow the pipeline panel.

        Args:
            delta: how far the pointer moved, in pixels.
        """
        self._layout.resize_pipeline(delta)
        self._apply_layout()

    # ------------------------------------------------------------------
    # Remembering the work
    # ------------------------------------------------------------------

    def _remember_workspace(self) -> None:
        """Write down what the editor is doing, for the next time it opens.

        Called after anything worth restoring changes. What is stored is a path,
        a pipeline and a few numbers, so writing it often costs nothing; and a
        failure to write is ignored, because losing the remembered session is a
        far smaller matter than interrupting someone's editing.
        """
        workspace.save(
            workspace.Workspace(
                image_path=self._current_path,
                pipeline=self._session.pipeline_text if self._session else "",
                layout=self._layout,
            )
        )

    async def _restore_workspace(self) -> None:
        """Reopen the photo and pipeline the editor was last left with."""
        remembered = self._workspace
        path = remembered.image_path

        if path is None or not path.is_file():
            # The photo has been moved or deleted since. Nothing is said about
            # it: the user did not ask for it to be reopened, so its absence is
            # not a failure they need telling about.
            return

        await self._open_image(path, announce_failure=False)

        if self._session is None or not remembered.pipeline:
            return

        try:
            invocations = parse_pipeline(remembered.pipeline)
        except PipelineDefinitionError:
            # A pipeline written by a version that knew a step this one does not.
            self._notify("The remembered pipeline could not be restored", is_error=True)
            return

        # Replayed through the ordinary path, so a step that has since changed
        # its parameters fails the same way it would if the user had added it.
        await self._change_pipeline(
            lambda session: _replay(session, invocations)
        )

    # ------------------------------------------------------------------
    # The work itself
    # ------------------------------------------------------------------

    async def _choose_and_open(self) -> None:
        """Show the open dialog and load whatever was chosen."""
        files = await self._file_picker.pick_files(
            dialog_title="Open an image",
            file_type=ft.FilePickerFileType.IMAGE,
            allowed_extensions=IMAGE_EXTENSIONS,
        )
        if not files:
            # The dialog was cancelled: leave the current image alone.
            return

        # On the web the picker returns the file's contents rather than a path,
        # so there is nothing on disk to open. The editor is a desktop app, but
        # the check keeps that from turning into a crash.
        chosen = files[0].path
        if chosen is None:
            self._notify("That file has no path on this platform", is_error=True)
            return

        await self._open_image(Path(chosen))

    async def _open_image(self, path: Path, announce_failure: bool = True) -> None:
        """Load an image from disk and start a fresh session on it.

        Args:
            path: the file to read.
            announce_failure: whether to tell the user if it cannot be read. It
                is silenced when reopening last time's photo, which the user did
                not ask for and need not be troubled about.
        """
        async with self._work_lock:
            await self._open_image_now(path, announce_failure)

    async def _open_image_now(self, path: Path, announce_failure: bool) -> None:
        """Load the image, with the right to touch the session already held.

        Args:
            path: the file to read.
            announce_failure: whether to tell the user if it cannot be read.
        """
        self._toolbar.set_busy(True)
        try:
            # Reading and decoding a large photo takes long enough to be worth
            # keeping off the interface's thread.
            image = await asyncio.to_thread(load_image, path)
        except (OSError, ValueError) as error:
            if announce_failure:
                self._notify(f"Could not open the image: {error}", is_error=True)
            return
        finally:
            self._toolbar.set_busy(False)

        self._session = EditingSession(image)
        self._current_path = path
        self._refresh()

    async def _apply_step(self, step_name: str) -> None:
        """Run one step and show the result.

        Args:
            step_name: the catalogue name of the step to run.
        """
        if self._session is None:
            self._notify("Open an image first", is_error=True)
            return

        async with self._work_lock:
            await self._apply_step_now(step_name)

    async def _apply_step_now(self, step_name: str) -> None:
        """Run one step, with the right to touch the session already held.

        Args:
            step_name: the catalogue name of the step to run.
        """
        session = self._session
        if session is None:
            return

        self._toolbar.set_busy(True)
        try:
            # The heavy lifting happens on a worker thread. `remove-background`
            # runs a neural network and takes seconds; the window has to stay
            # responsive throughout.
            await asyncio.to_thread(session.append, StepInvocation(step_name))
        except Exception as error:  # noqa: BLE001 - see below
            # A step can fail for reasons well outside this module's control: a
            # model that will not download, an unsupported parameter, OpenCV
            # refusing an image size. Whatever it is, the editor reports it and
            # carries on rather than taking the window down with it.
            self._notify(f"{step_name} failed: {error}", is_error=True)
            return
        finally:
            self._toolbar.set_busy(False)

        self._refresh()

    async def _change_pipeline(
        self, operation: Callable[[EditingSession], object]
    ) -> None:
        """Carry out one change to the pipeline, off the interface's thread.

        Moving, removing, reconfiguring and undoing all re-run part of the
        pipeline, and that part may contain a step that takes seconds. They
        therefore share one worker rather than each risking a frozen window, and
        one place to report a failure.

        Args:
            operation: what to do to the session. Taking the session as an
                argument rather than closing over it keeps the operation honest
                about what it is allowed to touch. Whatever it returns is
                ignored, so `undo` and the rest can share this one path.
        """
        async with self._work_lock:
            await self._change_pipeline_now(operation)

    async def _change_pipeline_now(
        self, operation: Callable[[EditingSession], object]
    ) -> None:
        """Carry out the change, with the right to touch the session already held.

        Args:
            operation: what to do to the session.
        """
        session = self._session
        if session is None:
            return

        self._toolbar.set_busy(True)
        try:
            await asyncio.to_thread(operation, session)
        except IndexError:
            # The pipeline changed under the request, which can happen if a step
            # was removed while its settings were open. Redrawing is enough.
            pass
        except Exception as error:  # noqa: BLE001 - see `_apply_step`
            self._notify(f"Could not change the pipeline: {error}", is_error=True)
        finally:
            self._toolbar.set_busy(False)

        self._refresh()

    async def _choose_and_save(self) -> None:
        """Show the save dialog and write the current image where asked."""
        session = self._session
        if session is None:
            return

        destination = await self._file_picker.save_file(
            dialog_title="Save the image",
            file_name=self.suggested_save_name(),
            allowed_extensions=IMAGE_EXTENSIONS,
        )
        if destination is None:
            # The dialog was cancelled.
            return

        path = Path(destination)
        try:
            # What gets written is the session's image: full resolution and full
            # transparency, not the scaled-down copy the canvas is showing.
            await asyncio.to_thread(save_image, session.current, path)
        except (OSError, ValueError) as error:
            self._notify(f"Could not save: {error}", is_error=True)
            return

        self._notify(f"Saved to {path.name}")

    def suggested_save_name(self) -> str:
        """Work out a sensible default file name for the save dialog.

        Returns:
            The opened file's name with an `-edited` suffix, or a generic default
            when nothing better is known.
        """
        if self._current_path is None:
            return DEFAULT_SAVE_NAME

        # PNG regardless of what was opened: a cut-out saved as JPEG would lose
        # its transparency without warning.
        return f"{self._current_path.stem}-edited.png"

    # ------------------------------------------------------------------
    # Redrawing
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Redraw every panel from the session.

        This is the only place the screen is updated after an edit. Going through
        one function means the canvas, the pipeline list and the toolbar buttons
        are always describing the same state.
        """
        session = self._session

        if session is None:
            self._canvas.show_placeholder()
            self._pipeline.show_steps((), "")
            self._toolbar.set_state(
                has_image=False, can_undo=False, can_redo=False, is_modified=False
            )
            self._toolbar.set_file_label("No image open")
            self._remember_workspace()
            return

        image = session.current
        self._canvas.show_image(preview.to_png_bytes(image))
        self._pipeline.show_steps(session.applied, session.pipeline_text)
        self._toolbar.set_state(
            has_image=True,
            can_undo=session.can_undo,
            can_redo=session.can_redo,
            is_modified=session.is_modified,
        )

        name = self._current_path.name if self._current_path else "untitled"
        self._toolbar.set_file_label(f"{name}  ·  {image.width}x{image.height}")

        # Every redraw follows a change worth remembering, so this is the one
        # place the work needs writing down from.
        self._remember_workspace()

    def _notify(self, message: str, is_error: bool = False) -> None:
        """Show a short message along the bottom of the window.

        Args:
            message: the text to show.
            is_error: True to colour it as a failure.
        """
        self._page.show_dialog(
            ft.SnackBar(
                content=ft.Text(value=message, color=theme.TEXT),
                bgcolor=theme.DANGER if is_error else theme.CARD_BACKGROUND,
                duration=MESSAGE_DURATION,
            )
        )


def _replay(
    session: EditingSession, invocations: Sequence[StepInvocation]
) -> None:
    """Rebuild a pipeline on a session, step by step.

    Used when reopening the work from last time. It goes through the session's
    ordinary `append`, so a remembered step whose parameters are no longer valid
    fails exactly as it would had the user just added it, rather than quietly
    producing a different picture.

    Args:
        session: the session to build the pipeline on.
        invocations: the steps to apply, in order.
    """
    for invocation in invocations:
        session.append(invocation)


def create_main(
    initial_image: Path | None = None, restore: bool = True
) -> Callable[[ft.Page], None]:
    """Build the page handler Flet calls when the window opens.

    Flet hands the handler nothing but the page, so anything else the editor
    needs to know has to be captured here, before the window exists.

    Args:
        initial_image: an image to open at start-up, if any.
        restore: whether to reopen the work from the last run.

    Returns:
        The handler to give to `ft.run`.
    """

    def main(page: ft.Page) -> None:
        """Build the editor on a freshly created page."""
        ImageEditorApp(page, initial_image, restore=restore)

    return main
