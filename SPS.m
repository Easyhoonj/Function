function [coresult,p_value]=SPS(spatial1,spatial2)
% SPS为Spatial Pattern Similarity (空间相关系数-空间场与空间场分布的相似度,并非是数值大小的相似)
% written by Jin Juncheng in 20261024 
% 另函数自带计算结果的显著性
% Input:
%   spatial1 - 空间场1(二维或一维)
%   spatial2 - 空间场2(二维或一维)
% Output:
%   coresult - 相关系数
%   p_value - p值,用于与显著性水平进行比较
%% First Step
% 二维转换为一维
if ~isvector(spatial1)
    spatial1=spatial1(:);
    spatial2=spatial2(:);
end
%% Second Step
% 这里的相关系数为Pearson相关系数
% 标记nan值,计算均值
mask_index1=isnan(spatial1);
mask_index2=isnan(spatial2);
n1=sum(mask_index1);   
n2=sum(mask_index2);   
if n1>=n2
    mask_index=mask_index1;
    n_nan=n1;
else
    mask_index=mask_index2;
    n_nan=n2;
end
% 两空间场同一位置数据都不为nan,才定义为有效值
spatial1(mask_index)=nan;
spatial2(mask_index)=nan;
spatial1_mean=mean(spatial1,"all","omitmissing");
spatial2_mean=mean(spatial2,"all","omitmissing");
coefficient1=sum((spatial1-spatial1_mean).*(spatial2-spatial2_mean),"all","omitmissing");
coefficient2=sqrt(sum((spatial1-spatial1_mean).^2,'all','omitmissing'));
coefficient3=sqrt(sum((spatial2-spatial2_mean).^2,'all','omitmissing'));
coresult=coefficient1/(coefficient2*coefficient3);
%% Third Step
% 统计显著性检验:使用t统计量+其CDF的双尾P值检验(这里由于不明确是正相关还是负相关,所以使用双尾检验;否则,可用单尾)
% H0假设:总体相关系数为0
% 此处有效样本数较为笼统(仅刨除二者nan值合集),专业处理可改善
N=numel(spatial1)-n_nan-2;
t_stat=coresult*sqrt(N/(1-coresult^2));
p_value=2*tcdf(-abs(t_stat),N);
end