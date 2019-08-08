function fdata=MeanfilterS(data,Iteratives)
nums=length(data);
fdata=data;
for ii=1:Iteratives
    %fdata=data;
    for jj=2:nums-1
        fdata(jj)=0.33333*(data(jj-1)+data(jj)+data(jj+1));
    end
    fdata(1)=0.5*(data(1)+data(2));
    fdata(nums)=0.5*(data(nums-1)+data(nums));
    data=fdata;
end

