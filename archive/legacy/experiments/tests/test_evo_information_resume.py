"""A configuration correction cannot silently change the scientific protocol."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoInputs import load_stage_config  # noqa: E402


def test_resume_only_corrects_numeric_progress_ratio():
    before=load_stage_config('stage2_information')
    after=load_stage_config('stage2_information_resume')
    assert before.validation_control.progress_floor=='0.5_times_recorded_reference'
    assert isinstance(after.validation_control.progress_floor,float)
    assert after.validation_control.progress_floor==.5
    assert after.training==before.training
    assert after.policy==before.policy
    assert after.validation_control.seed==before.validation_control.seed
    assert after.resume.rerun_training is False
