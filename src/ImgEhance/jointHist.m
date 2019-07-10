function [ JH ] = jointHist( IM1,IM2 )
%
%   IM1: Image  IM2:PCNN ([0,1]value)
[LEN, WID] = size(IM1);
JH = zeros(256,2);
for ii = 1:LEN
    for jj = 1:WID
        JH(IM1(ii,jj)+1,IM2(ii,jj)+1) = JH(IM1(ii,jj)+1,IM2(ii,jj)+1) + 1;
    end
end

end

