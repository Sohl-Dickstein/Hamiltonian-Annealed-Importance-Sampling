function [v] = getField(options,opt,default)
% extracts the field "opt" from the structure "options", or returns the
% value "default" if the field doesn't exist.  used to pass parameters into
% HAIS.m

    options = toUpper(options); % make fields case insensitive
    opt = upper(opt);
    if isfield(options,opt)
        if ~isempty(getfield(options,opt))
            v = getfield(options,opt);
        else
            v = default;
        end
    else
        v = default;
    end
end

function [o] = toUpper(o)
    if ~isempty(o)
        fn = fieldnames(o);
        for i = 1:length(fn)
            o = setfield(o,upper(fn{i}),getfield(o,fn{i}));
        end
    end
end

