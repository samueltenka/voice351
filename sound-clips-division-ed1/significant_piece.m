function [bool] = significant_piece(X)
    counter=0;
    for i=1:length(X)
        if X(i)~=0
            counter=counter+1;
        end
    end
    if counter>0.6*length(X)
        bool=1;
    else
        bool=0;
    end
end