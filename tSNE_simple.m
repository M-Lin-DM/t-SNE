function emb=tSNE_simple(dat, embDim, T, sigma)
% Author: Michael R. Lin 2019 Arizona State Univeristy
%perform tSNE without computing neighborhood size for each
%point. Each point uses an identicle neighborhood size sigma (width of radial
%basis function) in computing similarities to other points.
%INPUT: dat = [N x D] data matrix of N points
%embDim = embedding dimension
% T = number of iterations
% OUTPUT: embedding of points in lower dimensional space, row order is
% preserved

eta=.1;%learning rate
alpha=@(x) .6*exp(-.02*x);%momentum

%shape parameters for kernels:
fan=.4; %lower->heavier tails
bulge=1.7;% higher->pulls weight inward from tails into a bulge shape, thins tails

%Define similarity metric in embedding space:
q=@(vect,matrx) (1+norm_array(bsxfun(@minus,vect,matrx)).^bulge).^-fan;%similarity function in lower dim map

% VdCdy_ij(sim of u to all in highD, sim of u to all in emb, list of
% vectors pointing from all to u):
VdCdy_ij=@(Vu_p,Vu_q, j2u) bsxfun(@times, 4*(bsxfun(@times,(Vu_p'-Vu_q'), j2u)),  (1+norm_array(j2u).^bulge).^-fan);

% generate intial condition for points in the low dimensional map/representation
Y=normrnd(0,1,[size(dat,1),embDim]);%initial condition for gradient decent. these are the point in the low Dim space

% find pairwise similarities in the high Dim space
G=@(d,sig) exp(-d.^2/(2*sig^2));
% P=1-squareform(pdist(dat(:,1:end),'cosine'));%cosine similarity
P=G(squareform(pdist(dat(:,1:end),'euclidean')), sigma);
P=P-diag(diag(P));%set diagonal to 0
P=P./sum(P,2);%make it a probability distribution. Each row sums to 1

% Gradient Decent
strf=cell(T,1);
strf{1,1}=Y;
strf{2,1}=Y;%do not alter

for t=3:T
%     t
Q=squareform(pdist(strf{t-1,1},q));
Q= Q./sum(Q,2);%make each row a probability distribution

for u=1:size(dat,1) %cycle through each point in embedding space
    dyi=sum(VdCdy_ij(P(u,:), Q(u,:), strf{t-1,1}(u,:)-strf{t-1,1} ),1);%VdCdy_ij(sim of u to all in highD, sim of u to all in emb, list of vectors pointing from all to u)
    strf{t,1}(u,:)= strf{t-1,1}(u,:)-eta*dyi+alpha(t)*(strf{t-1,1}(u,:)-strf{t-2,1}(u,:));%move current point u down the gradient. add a momentum term too
end

%for plotting the process
t
% figure(3)
% plot(strf{t,1}(:,1),strf{t,1}(:,2),'k.')
% % plot3(strf{t,1}(:,1),strf{t,1}(:,2),strf{t,1}(:,3),'k.')
% % scatter3(strf{t,1}(:,1),strf{t,1}(:,2),strf{t,1}(:,3),5,colsp,'filled')
% axis equal
% box on
% % camorbit(100*(t/T),-10*(t/T))
% pause(0.01)

end
emb=strf{t,1};
end