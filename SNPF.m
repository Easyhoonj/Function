function [data_mean,data_anomaly,t_choose,v_choose,X_s_np_primitive,n_contribute,m_contribute]=SNPF(data,lat)
%% S/N-maximizing patterns filtering
% written by Jin Juncheng in 20261005
% firstly revised in 20261020

% data - 输入数据为四维矩阵(nLon,nLat,nTime,nMember).
% lat - 用于网格权重平均,更加贴近事实([nLat,1])
% 可以额外添加时空转换加速分解过程

% data_mean - 四维数组,使用需要squeeze
% data_anomaly - 数据距平处理结果
% t_choose - 选择的前M个PC时间系数,大小为[nMember,nTime,M]
% v_choose - 选择的前M个空间分布,大小为[nLon,nLat,M]
% X_s_np_primitive - 人为重构场
%% 第一步:数据预处理
% 各维度长度
[nLon,nLat,nTime,nMember]=size(data);
% 距平序列(MATLAB会自动沿维度为1的方向扩展)
data_mean=mean(mean(data,3,'omitmissing'),4,"omitmissing");
data_anomaly=data-data_mean;
% 设置陆地掩码nan,防止后续nan值扩散
mask_land=all(all(isnan(data_anomaly),4),3);
idx_ocean=find(~mask_land);
data_anomaly(isnan(data_anomaly))=0;

w=sqrt(cosd(lat));
w_reshape=reshape(w,1,nLat,1,1);
w_2d=repmat(w',nLon,1);
% 调整维度, n*ne*p, n为每个成员的时间维长度, ne为成员数, p为空间格点数.
data_shape=permute(data_anomaly.*w_reshape,[3,4,1,2]);
X=reshape(data_shape,nTime*nMember,nLon*nLat);
%% 第二步:集合平均协方差矩阵
for i=1:nMember
    Xi=permute(squeeze(data_anomaly(:,:,:,i).*w_reshape),[3,1,2]);
    Xi=reshape(Xi,nTime,nLon*nLat);
    Ci{i}=(Xi'*Xi)/(nTime-1);
end

C=sum(cat(3,Ci{:}),3,"omitmissing");
C_ensemble=C/nMember;
%% 第三步:分解特征向量以及特征值矩阵
% eig默认分解特征向量矩阵归一化处理
[A,sigma_var_matrix]=eig(C_ensemble);
S_var=diag(sigma_var_matrix);
% 对特征值进行排序
[sigma_var_sort,index]=sort(S_var,'descend');
sigma_var_matrix=sigma_var_matrix(index,index);
A=A(:,index);
%% 第四步
for i=1:length(S_var)
    ck(:,i)=X*A(:,i)/sqrt(sigma_var_sort(i));
end
%% 第五步: Sek=skek
% 取前N个EOF模态参与后续,用North判断法判断,大约75%~95%区间即可. N可取50~400, M可取2~20 
for i=1:length(S_var)-1
    ej=sigma_var_sort(i)*sqrt(2/length(S_var));
    if sigma_var_sort(i)-sigma_var_sort(i+1)<ej
        N=i;
    end
end
% 人为控制,防止N过大,导致量级爆炸
if N>225
    N=225;
elseif N<50
    error('N选取过少,不足以选取');
end
% 计算前N个模态方差贡献,单位为%
for i=1:N
    n_contribute(i)=sigma_var_sort(i)/sum(sigma_var_sort)*100;
end
% 求取ck集合平均
ck_primitive=reshape(ck,nTime,nMember,length(sigma_var_sort));
% 插曲:用于检测数组还原是否符合实际意义
m=10; t=3; k=1;
row=(m-1)*nTime+t;
val_ck=ck(row,k);
val_reshape=ck_primitive(t,m,k);
disp([val_ck,val_reshape])
% 求解
ck_ensemble=squeeze(sum(ck_primitive,2,'omitmissing')/nMember);
S=ck_ensemble(:,1:N)'*ck_ensemble(:,1:N);
% 求解特征向量及特征值矩阵
[e,s]=eig(S);
u=A(:,1:N)./(sqrt(sigma_var_sort(1:N)))'*e;
t=X*u;
%% 第六步: 排序提取M个模态
v=A(:,1:N).*(sqrt(sigma_var_sort(1:N)))'*e;
% 对上步所求特征值进行排序
s_var=diag(s);
[s_sort,index]=sort(s_var,'descend');
s=s(index,index);
e=e(:,index);
v=v(:,index);
t=t(:,index);
% 取前M个EOF模态参与后续,用North判断法判断,大约75%~95%区间即可
for i=1:length(s_var)-1
    ej=s_sort(i)*sqrt(2/length(s_var));
    if s_sort(i)-s_sort(i+1)<ej
        M=i;
    end
end
if M>11
    M=11;
elseif M<2
    error('M选取过少,不足以选取');
end
% 计算前M个模态方差贡献,单位为%
for i=1:M
    m_contribute(i)=s_sort(i)/sum(s_sort)*100;
end
%% 第七步: 取前M个模态还原数据信息(还原的为所有成员信息)
X_s_np=t(:,1:M)*v(:,1:M)';
% 还原维度信息,注意距平还原.
X_s_np_shape=reshape(X_s_np,nTime,nMember,nLon,nLat);
X_s_np_shape_anomaly=permute(X_s_np_shape,[3,4,1,2]);
X_s_np_shape_anomaly=X_s_np_shape_anomaly./w_reshape;
X_s_np_primitive=X_s_np_shape_anomaly+data_mean;
% 标准化t和v, v单位是data单位, t单位是标准差;归一化操作,便于不同模态t进行比较
sd=std(t(:,1:M),0,1);               
t(:,1:M)=t(:,1:M)./sd;
v(:,1:M)=v(:,1:M).*sd;             
t_choose=reshape(t(:,1:M),nMember,nTime,M);
v_choose=reshape(v(:,1:M),nLon,nLat,M);
for i=1:M
    tmp=v_choose(:,:,i);
    tmp(mask_land)=nan;
    v_choose(:,:,i)=tmp./w_2d;
end
end