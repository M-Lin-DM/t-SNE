function emb=tSNE_Prefab_similarityMatrix(P,yDim,T)
% Author: Michael R. Lin 2019 Arizona State Univeristy
% perform tSNE but using a similarity matrix P computed elsewhere. This is
% useful if you dont want to use tSNE's similarity function

%INPUT: P=predefined similarity matrix in original higher dimensional space
%yDim: dimension of embedding
%T: number of iterations
% OUTPUT: embedding of points in lower dimensional space

N = size(P,1);%number of dat points
%
eta=.18;%learning rate
alpha=@(x) 1*exp(-0.01*x);%momentum
% yDim=3;%dimension of space to embed data points into

%shape parameters for kernels:
fan=.7; %lower->heavier tails
bulge=2;% higher->pulls weight inward from tails into a bulge shape, thins tails

% q = similarity in LOW dim space 

%Define kernels for similarity measurement. i and j are vectors in some
%arbitrary dimension. p and q are both SIMILARITY, not DISTANCE functions. ie they grow as dist(i,j)-->0

%Define similarity metric in embedding space:
q=@(vect,matrx) (1+norm_array(bsxfun(@minus,vect,matrx)).^bulge).^-fan;%similarity function in lower dim map

VdCdy_ij=@(Vu_p,Vu_q, j2u) bsxfun(@times, 4*(bsxfun(@times,(Vu_p'-Vu_q'), j2u)),  (1+norm_array(j2u).^bulge).^-fan);
%VdCdy_ij(similarity of u to all in high D, similarity of u to all in low D, list of vectors pointing from all to u). (yi-yj) points j-->i.

% generate intial condition for points in the low dimensional map/representation
Y=normrnd(0,1,[N,yDim]);%initial condition for gradient decent. these are the point in the low Dim space

P=bsxfun(@rdivide, P, sum(P,2));%make it a probability distribution. Each row sums to 1

% Gradient Decent

state1=Y;
state2=Y;
state3=zeros(size(Y));
% colsp=jet(N);
tic
for t=1:T
    disp([num2str(t) ' out of ' num2str(T)])
    disp(['Time Elapsed: ' num2str(toc/3600) ' hours'])
    disp(['Time Remaining: ' num2str((toc/3600)*(T/t - 1)) ' hours'])%=(Total time - elapsed time)
Q=squareform(pdist(state2,q));%similarity matrix in embedding space
%(diagonal should already be 0)
Q=bsxfun(@rdivide, Q, sum(Q,2));%make it a probability distribution

if sum(sum(isnan(state3)))>0
return
end

for u=1:N %cycle through each point in embedding space
    dyi=sum(VdCdy_ij(P(u,:), Q(u,:), bsxfun(@minus,state2(u,:),state2) ),1);%VdCdy_ij(sim of u to all in highD, sim of u to all in emb, list of vectors pointing from all to u)
    state3(u,:)= state2(u,:)-eta*dyi+alpha(t)*(state2(u,:)-state1(u,:));%move current point u down the gradient. add a momentum term too
end
state1=state2;%order of these 2 statements is very important
state2=state3;

%     if mod(t,10)==0
%     figure(3)
% 
%     % plot(state3(:,1),state3(:,2),'ro')
%     plot3(state3(:,1),state3(:,2),state3(:,3),'k.')
% %     scatter3(state3(:,1),state3(:,2),state3(:,3),5,colsp,'filled')
%     % scatter3(state3(:,1),state3(:,2),state3(:,3),10,cdat,'filled')
%     axis equal
%     box on
%     camorbit(100*(t/T),-10*(t/T))
%     pause(0.01)
%     end
end
emb=state2; 
toc
end