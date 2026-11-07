"""Boundary resolution and leakage checks for single-position causal controls."""
import importlib.util
import unittest
from ddmi.models.positions import locate_positions

class PositionIndices(unittest.TestCase):
    def test_four_distinct_positions(self):
        text='[U]Question?[E][A]xx'
        offsets=[(0,3),(3,11),(11,12),(12,15),(15,18),(18,19),(19,20)]
        # Exact content covers Question? at [3,12), before assistant template.
        pos=locate_positions(text,'Question?',list(range(7)),offsets)
        self.assertEqual(pos,{'final_prefill':6,'previous_prefill':5,'two_back_prefill':4,'last_user_content':2})
    def test_no_assumed_constant_user_offset(self):
        text='HabcT12345';offsets=[(i,i+1) for i in range(len(text))]
        pos=locate_positions(text,'abc',list(range(len(text))),offsets)
        self.assertEqual(pos['last_user_content'],3)
        self.assertEqual(pos['final_prefill'],9)
    def test_rejects_truncated_content(self):
        with self.assertRaises(ValueError):locate_positions('HabcT12345','abc',[0,1,2,3],[(0,1),(1,2),(2,3),(5,6)])
    def test_rejects_boundary_straddling(self):
        with self.assertRaises(ValueError):locate_positions('HabcT12345','abc',list(range(6)),[(0,2),(2,3),(3,4),(4,5),(5,6),(6,7)])
    def test_rejects_repeated_content(self):
        with self.assertRaises(ValueError):locate_positions('abcabc','abc',list(range(6)),[(i,i+1) for i in range(6)])
    def test_rejects_identical_controls(self):
        with self.assertRaises(ValueError):locate_positions('Habc','abc',list(range(4)),[(i,i+1) for i in range(4)])

@unittest.skipIf(importlib.util.find_spec('torch') is None,'PyTorch is required for the tensor smoke tests')
class PositionHook(unittest.TestCase):
    def test_exactly_one_position_and_no_decode_leakage(self):
        import torch
        from ddmi.editing.measured import MeasuredIntervention
        for mode in ('position_final_fixed','position_previous_fixed','position_two_back_fixed','position_user_last_fixed'):
            for index in (0,3,7):
                hook=MeasuredIntervention('unused',[1.,0.,0.,0.],mode,4.)
                hook.set_prefill_position(index)
                x=torch.zeros((1,8,4),dtype=torch.float16)
                y=hook._hook(None,(),x)
                expected=torch.zeros_like(x);expected[0,index,0]=-4
                torch.testing.assert_close(y,expected,rtol=0,atol=0)
                decode=torch.ones((1,1,4),dtype=torch.float16)
                torch.testing.assert_close(hook._hook(None,(),decode),decode,rtol=0,atol=0)
                hook.reset()
                with self.assertRaises(ValueError):hook._hook(None,(),x)
    def test_callback_cannot_move_a_position_mid_decode(self):
        import torch
        from ddmi.editing.measured import MeasuredIntervention
        hook=MeasuredIntervention('unused',[1.,0.],'position_final_fixed',4.)
        hook.set_prefill_position(1);hook._hook(None,(),torch.zeros((1,2,2)))
        with self.assertRaises(ValueError):hook.set_prefill_position(0)

if __name__=='__main__':unittest.main()
