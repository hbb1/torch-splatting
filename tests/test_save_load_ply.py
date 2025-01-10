import unittest
import torch
import tempfile

from render import load_ply_model


class TestSaveLoad(unittest.TestCase):
    def test_save_load(self):
        plypath = '../result/test/splats-200.ply'

        with tempfile.NamedTemporaryFile(delete=True, suffix='.ply') as tmp_file:
            plytest = tmp_file.name

            model = load_ply_model(plypath)
            model.save_ply(plytest)
            model2 = load_ply_model(plytest)

            assert torch.equal(model._xyz, model2._xyz)
            assert torch.equal(model._features_dc, model2._features_dc)
            assert torch.equal(model._features_rest, model2._features_rest)
            assert torch.equal(model._scaling, model2._scaling)
            assert torch.equal(model._rotation, model2._rotation)
            assert torch.equal(model._opacity, model2._opacity)

            with open(plypath, 'rb') as file1, open(plytest, 'rb') as file2:
                self.assertEqual(file1.read(), file2.read(), 'Ply files are not equal')

if __name__ == '__main__':
    unittest.main()