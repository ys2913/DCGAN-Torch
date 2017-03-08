require 'nn'
local M = {}

local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  init'nn.SpatialConvolution'
end


function M.generator(noise_input, dis_filters, gen_filters, output)
	
	local BatchNormalization = nn.SpatialBatchNormalization
	local Convolution = nn.SpatialConvolution
	local FullConvolution = nn.SpatialFullConvolution

	local gen = nn.Sequential()
	gen:add(FullConvolution(noise_input, gen_filters * 8, 4, 4))
	gen:add(BatchNormalization(gen_filters * 8))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 8, gen_filters * 4, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 4))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 4, gen_filters * 2, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 2))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 2, gen_filters, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters, output, 4, 4, 2, 2, 1, 1))
	gen:add(nn.Tanh())
	--MSRinit(gen)
	return gen
end

function M.discriminator(noise_input, dis_filters, gen_filters, output)
	
	local BatchNormalization = nn.SpatialBatchNormalization
	local Convolution = nn.SpatialConvolution
	local FullConvolution = nn.SpatialFullConvolution

	local dis = nn.Sequential()

	dis:add(Convolution(output, dis_filters, 4, 4, 2, 2, 1, 1))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters, dis_filters * 2, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 2))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 2, dis_filters * 4, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 4))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 4, dis_filters * 8, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 8))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 8, 1, 4, 4))
	dis:add(nn.Sigmoid())

	dis:add(nn.View(1):setNumInputDims(3))
	--MSRinit(dis)
	return dis
end

function M.generator_128(noise_input, dis_filters, gen_filters, output)
	
	local BatchNormalization = nn.SpatialBatchNormalization
	local Convolution = nn.SpatialConvolution
	local FullConvolution = nn.SpatialFullConvolution

	local gen = nn.Sequential()
	gen:add(FullConvolution(noise_input, gen_filters * 16, 4, 4))
	gen:add(BatchNormalization(gen_filters * 16))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 16, gen_filters * 8, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 8))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 8, gen_filters * 4, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 4))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 4, gen_filters * 2, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 2))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 2, gen_filters, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters))
	gen:add(nn.ReLU(true))

	gen:add(FullConvolution(gen_filters, output, 4, 4, 2, 2, 1, 1))
	gen:add(nn.Tanh())

	return gen
end

function M.discriminator_128(noise_input, dis_filters, gen_filters, output)
	
	local BatchNormalization = nn.SpatialBatchNormalization
	local Convolution = nn.SpatialConvolution
	local FullConvolution = nn.SpatialFullConvolution

	local dis = nn.Sequential()

	dis:add(Convolution(output, dis_filters, 4, 4, 2, 2, 1, 1))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters, dis_filters * 2, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 2))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 2, dis_filters * 4, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 4))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 4, dis_filters * 8, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 8))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 8, dis_filters * 16, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 16))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 16, 1, 4, 4))
	dis:add(nn.Sigmoid())

	dis:add(nn.View(1):setNumInputDims(3))

	return dis
end

function M.generator_256(noise_input, dis_filters, gen_filters, output)
	
	local BatchNormalization = nn.SpatialBatchNormalization
	local Convolution = nn.SpatialConvolution
	local FullConvolution = nn.SpatialFullConvolution

	local gen = nn.Sequential()

	gen:add(FullConvolution(noise_input, gen_filters * 32, 4, 4))
	gen:add(BatchNormalization(gen_filters * 32))
	gen:add(nn.ReLU(true))

	gen:add(FullConvolution(noise_input * 32, gen_filters * 16, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 16))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 16, gen_filters * 8, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 8))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 8, gen_filters * 4, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 4))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 4, gen_filters * 2, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters * 2))
	gen:add(nn.ReLU(true))
	
	gen:add(FullConvolution(gen_filters * 2, gen_filters, 4, 4, 2, 2, 1, 1))
	gen:add(BatchNormalization(gen_filters))
	gen:add(nn.ReLU(true))

	gen:add(FullConvolution(gen_filters, output, 4, 4, 2, 2, 1, 1))
	gen:add(nn.Tanh())

	return gen
end

function M.discriminator_256(noise_input, dis_filters, gen_filters, output)
	
	local BatchNormalization = nn.SpatialBatchNormalization
	local Convolution = nn.SpatialConvolution
	local FullConvolution = nn.SpatialFullConvolution

	local dis = nn.Sequential()

	dis:add(Convolution(output, dis_filters, 4, 4, 2, 2, 1, 1))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters, dis_filters * 2, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 2))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 2, dis_filters * 4, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 4))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 4, dis_filters * 8, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 8))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 8, dis_filters * 16, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 16))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 16, dis_filters * 32, 4, 4, 2, 2, 1, 1))
	dis:add(BatchNormalization(dis_filters * 32))
	dis:add(nn.LeakyReLU(0.2, true))

	dis:add(Convolution(dis_filters * 32, 1, 4, 4))
	dis:add(nn.Sigmoid())

	dis:add(nn.View(1):setNumInputDims(3))

	return dis
end

return M
