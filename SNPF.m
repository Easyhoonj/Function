function [data_mean,data_anomaly,t_choose,v_choose,X_s_np_primitive,n_contribute,m_contribute,SN_ratio]=SNPF(data,lat)
%% S/N-maximizing patterns filtering
% written by Jin Juncheng in 20261005
% data - 输入数据为四维矩阵(nLon,nLat,nTime,nMember)
% lat - 用于网格权重平均,更加贴近事实([nLat,1])
% 可以额外添加时空转换加速分解过程
% data_mean - 四维数组,使用需要squeeze
% data_anomaly - 数据距平处理结果
% t_choose - 选择的前M个PC时间系数,大小为[nMember,nTime,M]
% v_choose - 选择的前M个空间分布,大小为[nLon,nLat,M]
% X_s_np_primitive - 人为重构场
% n_contribute - 第一步所挑选前N个空间模态对应的方差贡献
% m_contribute - 第二步所挑选前M个空间模态对应的方差贡献
% SN_ratio - 第k个最终模态对应的信噪比
%% 第一步:数据预处理
% 各维度长度
[nLon,nLat,nTime,nMember]=size(data);
% 距平序列(MATLAB会自动沿维度为1的方向扩展)
data_mean=mean(mean(data,3,'omitmissing'),4,"omitmissing");
data_anomaly=data-data_mean;
% 设置陆地掩码nan,防止后续nan值扩散
mask_land=all(all(isnan(data_anomaly),4),3);
data_anomaly(isnan(data_anomaly))=0;
% 网格权重(EOF分析协方差)
w=sqrt(cosd(lat));
w_reshape=reshape(w,1,nLat,1,1);
w_2d=repmat(w',nLon,1);
% 调整维度, n*ne*p, n为每个成员的时间维长度, ne为成员数, p为空间格点数
data_shape=permute(data_anomaly.*w_reshape,[3,4,1,2]);
X=reshape(data_shape,nTime*nMember,nLon*nLat);
%% 第二步:集合平均协方差矩阵
for i=1:nMember
    Xi=permute(squeeze(data_anomaly(:,:,:,i).*w_reshape),[3,1,2]);
    Xi=reshape(Xi,nTime,nLon*nLat);
    Ci{i}=(Xi'*Xi)/(nTime-1);
end
% 求解集合平均
C=sum(cat(3,Ci{:}),3,"omitmissing");
C_ensemble=C/nMember;
%% 第三步:分解特征向量以及特征值矩阵
% eig默认分解特征向量矩阵归一化处理
% 特征值求解前提矩阵为方阵
[A,sigma_var_matrix]=eig(C_ensemble);
S_var=diag(sigma_var_matrix);
% 对特征值进行排序
[sigma_var_sort,index]=sort(S_var,'descend');
sigma_var_matrix=sigma_var_matrix(index,index);
A=A(:,index);
%% 第四步: 求解主成分PCn序列并归一化
% 防止病态、失真、选错模态,小模态信息在数值上被彻底淹没
% 同时归一化后,使得各PC能够进行比较(否则因为PC量级不同,可能只有PC大的有明显起伏,其余为平线,处理时也容易导致误差)
for i=1:length(S_var)
    ck(:,i)=X*A(:,i)/sqrt(sigma_var_sort(i));
end
%% 第五步: Sek=skek
% 取前N个EOF模态参与后续,用North判断法判断,大约75%~95%区间即可
% 前提假设各样本相互独立,所以有效样本与样本数目可认为等同
% 经验性取法: N可取50~400(取中位数), M可取2~20 
N=[];
for i=1:length(S_var)-1
    ej=sigma_var_sort(i)*sqrt(2/(nTime*nMember));
    if sigma_var_sort(i)-sigma_var_sort(i+1)<ej
        N=i;
        break;
    end
end
if isempty(N)
    N=median(length(S_var));
end
% 人为控制,防止N过大,导致后续计算时量级爆破
% 此处经验性的取中位数
if N>225
    N=225;
end
% 计算前N个模态方差贡献,单位为%
for i=1:N
    n_contribute(i)=sigma_var_sort(i)/sum(sigma_var_sort)*100;
end
% 求取ck集合平均
ck_primitive=reshape(ck,nTime,nMember,length(sigma_var_sort));
% 经过检验,数组还原仍旧符合实际物理意义,并没有发生信息错位现象
% 求解: 这里的ck是由前面的距平阵变形而来,无需再次距平(类似Ci)
% ck'*ck为离差平方和矩阵,要除以n-1(无偏),得到协方差阵
ck_ensemble=squeeze(sum(ck_primitive,2,'omitmissing')/nMember);
S=ck_ensemble(:,1:N)'*ck_ensemble(:,1:N)/(nTime-1);
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
M=[];
for i=1:length(s_var)-1
    ej=s_sort(i)*sqrt(2/(nTime*nMember));
    if s_sort(i)-s_sort(i+1)<ej
        M=i;
        break;
    end
end
% 同N的处理方法,防止过长M取值,导致量级爆破
if isempty(M)
    M=N;
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
% 计算信噪比
for i=1:M
    SN_ratio(i)=s_sort(i)/(1-s_sort(i));
end
%% 第七步: 取前M个模态还原数据信息(还原的为所有成员信息)
X_s_np=t(:,1:M)*v(:,1:M)';
% 还原维度信息,注意距平还原.
X_s_np_shape=reshape(X_s_np,nTime,nMember,nLon,nLat);
X_s_np_shape_anomaly=permute(X_s_np_shape,[3,4,1,2]);
X_s_np_shape_anomaly=X_s_np_shape_anomaly./w_reshape;
X_s_np_primitive=X_s_np_shape_anomaly+data_mean;
% 标准化t和v, v单位与data单位相同, t单位是标准差;归一化操作,便于不同模态t进行比较(这一步主要是为了绘图,前面则为了保持变换一致以及标准化PC)
sd=std(t(:,1:M),0,1);               
t(:,1:M)=t(:,1:M)./sd;
v(:,1:M)=v(:,1:M).*sd;             
t_choose=reshape(t(:,1:M),nTime,nMember,M);
v_choose=reshape(v(:,1:M),nLon,nLat,M);
for i=1:M
    tmp=v_choose(:,:,i);
    tmp(mask_land)=nan;
    v_choose(:,:,i)=tmp./w_2d;
end
end