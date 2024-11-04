function net = spdnet_init_afew(opts)
% spdnet_init initializes a spdnet
rng('default');
rng(0) ;
opts.layernum = 3;
Winit = cell(opts.layernum,1);
opts.datadim = [length(opts.train(1).X), 250, 125, length(opts.train(1).Y)];



for iw = 1 : opts.layernum
    A = rand(opts.datadim(iw));
    [U1, S1, V1] = svd(A * A');
    Winit{iw} = U1(:,1:opts.datadim(iw+1));
end


net.layers = {} ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{3}) ;
net.layers{end+1} = struct('type', opts.loss_function) ;






