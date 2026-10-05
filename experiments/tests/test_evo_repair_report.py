"""Offline audit must retain exact decisions while tolerating floating reductions."""
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

import pytest

from evo_repair_report import verify_nomination


def test_only_machine_precision_progress_roundoff_is_permitted():
    expected=dict(index=0,eligible=[],imagined_risk=[0.,.1],imagined_progress=[1.,2.])
    actual=copy.deepcopy(expected)
    actual['imagined_progress'][0]+=8e-16
    assert 0<verify_nomination(actual,expected)<1e-14
    actual['imagined_progress'][0]+=.000001
    with pytest.raises(AssertionError):
        verify_nomination(actual,expected)


def test_controller_choice_and_eligibility_remain_exact():
    expected=dict(index=0,eligible=[],imagined_risk=[0.,.1],imagined_progress=[1.,2.])
    for key,value in [('index',1),('eligible',[1]),('imagined_risk',[1e-16,.1])]:
        actual=copy.deepcopy(expected)
        actual[key]=value
        with pytest.raises(AssertionError):
            verify_nomination(actual,expected)
