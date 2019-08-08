function  a=ray_burstsampling1(LLs,three_vs);
nx=length(LLs);
a=0;
for i=1:nx
    if LLs(i)<three_vs
        break
    else
        a=a+1;
    end
end