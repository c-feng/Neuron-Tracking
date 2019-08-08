function ss=InnerEnergeComput(Line1,Line2)
if isempty(Line2)==1
    ss=0;
else
    [~,Pver]=ProbConnect(Line1,Line2);
    ss=Pver-1;    
end





