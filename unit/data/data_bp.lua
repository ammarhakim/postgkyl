-- Gkyl ------------------------------------------------------------------------
--
-- Dummy file creator for Postgkyl unit test
-- Author: pc
--------------------------------------------------------------------------------

local Grid = require "Grid"
local DataStruct = require "DataStruct"
local Time = require "Lib.Time"
local Basis = require "Basis"
local Updater = require "Updater"

local phaseGrid = Grid.RectCart {
   lower = {-1.0, -1.0, -1.0, -1.0, -1.0},
   upper = {1.0, 1.0, 1.0, 1.0, 1.0},
   cells = {2, 2, 2, 2, 2},
}

for polyOrder = 1, 4 do
   local basis = {'ms', 'mo'}
   for i, b in ipairs(basis) do
      -- basis functions
      if b == 'ns' then
	 phaseBasis = Basis.CartNodalSerendipity { ndim = phaseGrid:ndim(),
						   polyOrder = polyOrder }
      elseif b == 'ms' then
	 phaseBasis = Basis.CartModalSerendipity { ndim = phaseGrid:ndim(),
						   polyOrder = polyOrder }
      elseif b == 'mo' then
	 phaseBasis = Basis.CartModalMaxOrder { ndim = phaseGrid:ndim(),
						polyOrder = polyOrder }
      else
	 error('Invalid basis')
      end
      
      -- field
      local distf = DataStruct.Field {
	 onGrid = phaseGrid,
	 numComponents = phaseBasis:numBasis(),
	 ghost = {0, 0},
      }
      
      -- updater to initialize distribution function
      local project = Updater.ProjectOnBasis {
	 onGrid = phaseGrid,
	 basis = phaseBasis,
	 evaluate = function (t, xn)
	    return 1
	 end
      }
      project:advance(0.0, 0.0, {}, {distf})
      
      distf:write(string.format("d_%d_%s_p_%d.bp", phaseGrid:ndim(), b, polyOrder), 0.0)
   end
end
