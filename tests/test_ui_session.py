"""Tests for the editing session.

The session is where the pipeline is assembled and reassembled, so it carries the
behaviour most worth pinning down: that order matters, that a change halfway
along rebuilds what follows it, and that anything done can be undone. It has no
Flet in it, which is what lets these tests run without a window.
"""

import numpy as np
import pytest
from conftest import gray_image, solid_image

from pixel.dsl import StepInvocation
from pixel.errors import InvalidParameterValueError, UnknownStepError
from pixel.ui.session import EditingSession


class TestFreshSession:
    """A session that has just been opened is unmodified."""

    def test_the_current_image_is_the_source(self) -> None:
        image = solid_image((10, 20, 30))

        session = EditingSession(image)

        assert session.current is image

    def test_the_pipeline_is_empty(self) -> None:
        session = EditingSession(solid_image((10, 20, 30)))

        assert session.applied == ()
        assert not session.can_undo
        assert not session.is_modified
        assert session.pipeline_text == ""


class TestAppending:
    """Adding a step must change the picture and record what was done."""

    def test_the_image_changes(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))

        session.append(StepInvocation("grayscale"))

        rgb = session.current.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_the_step_is_recorded(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))

        session.append(StepInvocation("grayscale"))

        assert [step.invocation.name for step in session.applied] == ["grayscale"]
        assert session.can_undo
        assert session.is_modified

    def test_the_source_is_left_untouched(self) -> None:
        # Reset depends on this: the original has to survive every edit.
        image = solid_image((200, 100, 50))
        session = EditingSession(image)

        session.append(StepInvocation("invert"))

        assert np.array_equal(session.source.rgb, image.rgb)

    def test_steps_stack_on_each_other(self) -> None:
        session = EditingSession(solid_image((0, 0, 0)))

        session.append(StepInvocation("invert"))
        session.append(StepInvocation("invert"))

        # Two inversions cancel out, which proves the second step ran on the
        # result of the first rather than on the original.
        assert np.all(session.current.rgb == 0)
        assert len(session.applied) == 2

    def test_parameters_are_honoured(self) -> None:
        session = EditingSession(solid_image((10, 20, 30), width=100, height=50))

        session.append(StepInvocation("resize", {"scale": "0.5"}))

        assert (session.current.width, session.current.height) == (50, 25)

    def test_an_unknown_step_leaves_the_session_alone(self) -> None:
        session = EditingSession(solid_image((10, 20, 30)))

        with pytest.raises(UnknownStepError):
            session.append(StepInvocation("nonexistent"))

        assert session.applied == ()
        assert not session.can_undo


class TestMoving:
    """Reordering must change the result, because order is meaningful."""

    def test_moving_changes_the_picture(self) -> None:
        # Posterising then inverting is not the same as inverting then
        # posterising: the quantisation lands on different levels.
        session = EditingSession(gray_image(100))
        session.append(StepInvocation("posterize", {"levels": "3"}))
        session.append(StepInvocation("invert"))
        before = session.current.rgb.copy()

        session.move(0, 1)

        assert not np.array_equal(session.current.rgb, before)

    def test_the_order_shown_follows_the_move(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))

        session.move(1, 0)

        assert session.pipeline_text == "invert | grayscale"

    def test_moving_a_step_to_its_own_place_does_nothing(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        undo_depth_before = session.can_undo

        session.move(0, 0)

        # Nothing happened, so nothing should have been recorded to undo either:
        # otherwise undo would appear to do nothing when pressed.
        assert session.can_undo == undo_depth_before
        assert session.pipeline_text == "grayscale"

    def test_a_destination_past_the_end_lands_on_the_end(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))

        session.move(0, 99)

        assert session.pipeline_text == "invert | grayscale"

    def test_a_destination_before_the_start_lands_on_the_start(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))

        session.move(1, -5)

        assert session.pipeline_text == "invert | grayscale"

    def test_moving_a_step_that_is_not_there_is_refused(self) -> None:
        session = EditingSession(solid_image((10, 20, 30)))

        with pytest.raises(IndexError):
            session.move(3, 0)


class TestRemoving:
    """Any step must be removable, not only the last one."""

    def test_a_step_in_the_middle_can_be_removed(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))
        session.append(StepInvocation("posterize"))

        session.remove_at(1)

        assert session.pipeline_text == "grayscale | posterize"

    def test_the_picture_is_rebuilt_without_it(self) -> None:
        session = EditingSession(gray_image(80))
        session.append(StepInvocation("invert"))
        only_invert = session.current.rgb.copy()
        session.append(StepInvocation("posterize", {"levels": "2"}))

        session.remove_at(1)

        assert np.array_equal(session.current.rgb, only_invert)

    def test_removing_the_only_step_returns_the_original(self) -> None:
        image = solid_image((200, 100, 50))
        session = EditingSession(image)
        session.append(StepInvocation("invert"))

        session.remove_at(0)

        assert np.array_equal(session.current.rgb, image.rgb)
        assert not session.is_modified

    def test_removing_a_step_that_is_not_there_is_refused(self) -> None:
        session = EditingSession(solid_image((10, 20, 30)))

        with pytest.raises(IndexError):
            session.remove_at(0)


class TestReconfiguring:
    """A step's parameters must be changeable after it has been applied."""

    def test_new_parameters_change_the_result(self) -> None:
        session = EditingSession(solid_image((10, 20, 30), width=100, height=100))
        session.append(StepInvocation("resize", {"scale": "0.5"}))

        session.replace_at(0, StepInvocation("resize", {"scale": "0.25"}))

        assert (session.current.width, session.current.height) == (25, 25)

    def test_the_change_shows_in_the_pipeline_text(self) -> None:
        session = EditingSession(gray_image(120))
        session.append(StepInvocation("posterize"))

        session.replace_at(0, StepInvocation("posterize", {"levels": "8"}))

        assert session.pipeline_text == "posterize:levels=8"

    def test_later_steps_are_rebuilt_on_the_new_result(self) -> None:
        session = EditingSession(solid_image((10, 20, 30), width=100, height=100))
        session.append(StepInvocation("resize", {"scale": "0.5"}))
        session.append(StepInvocation("grayscale"))

        session.replace_at(0, StepInvocation("resize", {"scale": "0.25"}))

        # The greyscale step must have re-run on the newly sized image.
        assert (session.current.width, session.current.height) == (25, 25)
        rgb = session.current.rgb
        assert np.array_equal(rgb[:, :, 0], rgb[:, :, 2])

    def test_a_bad_parameter_leaves_the_pipeline_alone(self) -> None:
        session = EditingSession(gray_image(120))
        session.append(StepInvocation("blur", {"radius": "2"}))

        with pytest.raises(InvalidParameterValueError):
            session.replace_at(0, StepInvocation("blur", {"radius": "lots"}))

        assert session.pipeline_text == "blur:radius=2"

    def test_reconfiguring_a_step_that_is_not_there_is_refused(self) -> None:
        session = EditingSession(solid_image((10, 20, 30)))

        with pytest.raises(IndexError):
            session.replace_at(0, StepInvocation("grayscale"))


class TestUndo:
    """Undo must take back the last change, whatever kind of change it was."""

    def test_it_takes_back_an_added_step(self) -> None:
        session = EditingSession(solid_image((0, 0, 0)))
        session.append(StepInvocation("invert"))

        assert session.undo()

        assert np.all(session.current.rgb == 0)
        assert session.applied == ()

    def test_it_takes_back_a_move(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))
        session.move(1, 0)

        session.undo()

        assert session.pipeline_text == "grayscale | invert"

    def test_it_takes_back_a_removal(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))
        session.remove_at(0)

        session.undo()

        assert session.pipeline_text == "grayscale | invert"

    def test_it_takes_back_a_parameter_change(self) -> None:
        session = EditingSession(gray_image(120))
        session.append(StepInvocation("posterize", {"levels": "4"}))
        session.replace_at(0, StepInvocation("posterize", {"levels": "9"}))

        session.undo()

        assert session.pipeline_text == "posterize:levels=4"

    def test_it_takes_back_a_reset(self) -> None:
        # Reset is the most destructive action in the editor, so it is the one
        # most worth being able to take back.
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))
        session.reset()

        session.undo()

        assert session.pipeline_text == "grayscale | invert"

    def test_the_picture_follows_the_undo(self) -> None:
        session = EditingSession(gray_image(80))
        session.append(StepInvocation("invert"))
        after_invert = session.current.rgb.copy()
        session.append(StepInvocation("posterize", {"levels": "2"}))

        session.undo()

        assert np.array_equal(session.current.rgb, after_invert)

    def test_undoing_an_untouched_session_reports_it_did_nothing(self) -> None:
        session = EditingSession(solid_image((10, 20, 30)))

        assert not session.undo()

    def test_several_changes_are_undone_one_at_a_time(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))
        session.move(1, 0)

        session.undo()
        session.undo()

        assert session.pipeline_text == "grayscale"


class TestReset:
    """Reset must go all the way back to the opened image."""

    def test_it_restores_the_source(self) -> None:
        image = solid_image((200, 100, 50))
        session = EditingSession(image)
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))

        session.reset()

        assert np.array_equal(session.current.rgb, image.rgb)
        assert session.applied == ()
        assert not session.is_modified

    def test_resetting_an_untouched_session_is_harmless(self) -> None:
        session = EditingSession(solid_image((10, 20, 30)))

        session.reset()

        assert not session.is_modified
        # Nothing changed, so there is nothing to undo either.
        assert not session.can_undo


class TestPipelineText:
    """The session must be able to describe itself in command line syntax."""

    def test_it_lists_the_steps_in_order(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))

        assert session.pipeline_text == "grayscale | invert"

    def test_it_includes_the_parameters(self) -> None:
        session = EditingSession(solid_image((10, 20, 30), width=100, height=50))

        session.append(StepInvocation("resize", {"scale": "0.5"}))

        assert session.pipeline_text == "resize:scale=0.5"

    def test_what_it_produces_can_be_run_from_the_command_line(self) -> None:
        # The text is offered to the user to copy into a terminal, so it has to
        # parse back into the very same steps.
        from pixel import build_pipeline

        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("posterize", {"levels": "6"}))

        rebuilt = build_pipeline(session.pipeline_text)

        assert list(rebuilt.iter_step_names()) == ["grayscale", "posterize"]


class TestResultsStayInStep:
    """The cached results must never drift from the pipeline that produced them."""

    def test_every_step_carries_its_own_result(self) -> None:
        session = EditingSession(solid_image((200, 100, 50)))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("invert"))
        session.move(1, 0)
        session.remove_at(1)

        applied = session.applied

        assert len(applied) == 1
        # The one surviving step's recorded result must be the current picture.
        assert np.array_equal(applied[-1].result.rgb, session.current.rgb)

    def test_the_result_matches_running_the_same_pipeline_from_scratch(self) -> None:
        # Whatever route the user took to build it, the picture must equal what
        # the same pipeline would produce in one go from the command line.
        from pixel import build_pipeline

        image = solid_image((200, 100, 50), width=40, height=40)
        session = EditingSession(image)
        session.append(StepInvocation("invert"))
        session.append(StepInvocation("grayscale"))
        session.append(StepInvocation("posterize", {"levels": "5"}))
        session.move(2, 0)
        session.remove_at(2)

        straight = build_pipeline(session.pipeline_text).run(image)

        assert np.array_equal(session.current.data, straight.final_image.data)
