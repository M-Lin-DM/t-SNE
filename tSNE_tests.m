%% test t-SNE on swissroll data set
% Y=normrnd(0,1,[size(dat,1),2]);%initial condition for gradient decent. these are the point in the low Dim space
%
% dat=load('swissroll'); dat=dat.dat; %import this data to compare the different graph creation rules 
figure(1)
% plot3(dat(:, 1), dat(:, 2), dat(:, 3), '.')

scatter3(dat(:, 1), dat(:, 2), dat(:, 3), 10, 1:size(dat,1),'filled'); axis equal
%%
tic
emb=tSNE_simple(dat,2,100, 1.5);
% [emb, sigma]=tSNE_perplexity(dat,2,100, 1.95);
toc
%%
figure(2)
% plot(emb(:,1), emb(:,2),'k.')
scatter(emb(:, 1), emb(:, 2), 10, 1:size(dat,1),'filled')
% scatter(emb(:, 1), emb(:, 2), 10, dat(:,3),'filled')
axis equal
% title('perplexity = 1.95', 'fontsize', 14)
%%
P=squareform(pdist(dat,'euclidean'));
G=@(d,sig) exp(-d.^2/(2*sig^2));
W=G(P,1.5);
emb=tSNE_Prefab_similarityMatrix(W,2,100);