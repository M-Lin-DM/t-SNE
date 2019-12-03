function [proj, princ]=PCA_simp(dat,dim)
%INPUT: dat= data matrix. Rows correspond to observations or data points.
%Columns are variables/features/dimensions
%dim=desired dimensionality of projected data 
%OUTPUT: proj=data matrix in lower dimensional space
u=mean(dat,1); 
D=size(dat,2);%dimensionality of input data
dat_cen=bsxfun(@minus,dat,u);%(subtract mean.) This centers data about origin in each dimension
Covmat=dat_cen'*dat_cen;
[V,~] = eig(Covmat);%[V]=[D x D] is  produces a diagonal matrix D of generalized
%     eigenvalues and a matrix V whose columns are the
%     corresponding eigenvectors (of length D). These eigenvectors are
%     ORTHONORMAL (so we dont need to multiply by D)
%
%KEY STEPS:
%now choose which eigvects to project onto
princ=V(:,D-dim+1:end)';%[princ]=[dim x D]= the dim rightmost (i.e. largest) principle components, transposed. if using 3 prin comps, order matters
proj=(princ*dat')';%[proj]=([dim x D]*[D x #points])'=[#points x dim]. proj is the data projected into the basis of the chosen principle components. If n principle components are chosen, proj is the data in n-D
end