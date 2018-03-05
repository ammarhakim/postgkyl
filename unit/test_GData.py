#!/usr/bin/env python
import unittest
import numpy as np
import postgkyl as pg

class TestGData(unittest.TestCase):
    def test_EmptyFile(self):
        data = pg.GData()

        nd = data.getNumDims()
        assert(isinstance(nd, int))
        self.assertEqual(nd, 0)

        nc = data.getNumComps()
        assert(isinstance(nc, int))
        self.assertEqual(nc, 0)

        lo, up = data.getBounds()
        assert(isinstance(lo, np.ndarray))
        assert(isinstance(up, np.ndarray))
        self.assertEqual(len(lo), 0)
        self.assertEqual(len(up), 0)

        nc = data.getNumCells()
        assert(isinstance(nc, np.ndarray))
        self.assertEqual(len(nc), 0)
        
    def test_LoadBpFrame(self):
        data = pg.GData('data/frame_0.bp')

        nd = data.getNumDims()
        assert(isinstance(nd, int))
        self.assertEqual(nd, 2)

        nc = data.getNumComps()
        assert(isinstance(nc, int))
        self.assertEqual(nc, 8)

        lo, up = data.getBounds()
        assert(isinstance(lo, np.ndarray))
        assert(isinstance(up, np.ndarray))
        self.assertEqual(len(lo), 2)
        self.assertEqual(len(up), 2)

        nc = data.getNumCells()
        assert(isinstance(nc, np.ndarray))
        assert(isinstance(nc[0], np.int64))
        assert(isinstance(nc[1], np.int64))
        self.assertEqual(nc[0], 64)
        self.assertEqual(nc[1], 32)

        grid = data.peakGrid()
        assert(isinstance(grid, list))
        self.assertEqual(len(grid), 2)
        assert(isinstance(grid[0], np.ndarray))
        assert(isinstance(grid[1], np.ndarray))
        self.assertEqual(grid[0].shape, (64,))
        self.assertEqual(grid[1].shape, (32,))

        values = data.peakValues()
        assert(isinstance(values, np.ndarray))
        self.assertEqual(values.shape, (64, 32, 8))

    def test_LoadBpHistory(self):
        data = pg.GData('data/hist_')

        nd = data.getNumDims()
        assert(isinstance(nd, int))
        self.assertEqual(nd, 1)

        nc = data.getNumComps()
        assert(isinstance(nc, int))
        self.assertEqual(nc, 8)

        lo, up = data.getBounds()
        assert(isinstance(lo, np.ndarray))
        assert(isinstance(up, np.ndarray))
        self.assertEqual(len(lo), 1)
        self.assertEqual(len(up), 1)

        nc = data.getNumCells()
        assert(isinstance(nc, np.ndarray))
        assert(isinstance(nc[0], np.int64))
        self.assertEqual(nc[0], 7641)

        grid = data.peakGrid()
        assert(isinstance(grid, list))
        self.assertEqual(len(grid), 1)
        assert(isinstance(grid[0], np.ndarray))
        self.assertEqual(grid[0].shape, (7641,))

        values = data.peakValues()
        assert(isinstance(values, np.ndarray))
        self.assertEqual(values.shape, (7641, 8))

    def test_Info(self):
        data = pg.GData('data/frame_0.bp')
        info = data.info() 

    def test_WriteBp(self):
        pass

    def test_WriteTxt(self):
        pass

if __name__ == '__main__':
    unittest.main()
