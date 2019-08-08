function xx=probdist_corr(dist,corr)

if abs(corr)>0.95
    a1=1;
else
    a1=exp(-(abs(corr)-0.95)^2/0.5);
end

if dist<3
    b1=1;
else
    b1=exp(-(dist-3)^2/5);
end
xx=a1*b1;






