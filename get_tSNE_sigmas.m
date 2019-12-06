function sigma = get_tSNE_sigmas(dat, perp_targ)
% Author: Michael R. Lin 2019 Arizona State Univeristy
%Input: dat = [N x D] data matrix in D dimensional space
%perp_targ = perplexity target in (1,2]. higher value requires a higher entropy (H) of the similarities of one point to all
%others. Higher entropy implies greater ability to capture large scale structures in the original data manifold. This seems to mean using larger 
%neigborhood sizes (sigma) in the radial basis kernel. 

N=size(dat,1);
% perp_targ=1.6;%=2^H. MUST BE WITHIN (1,2] since H in [0,1] perplexity target perp(p_i)=2^H(p_i) where H(p_i)=-sum_i(p_(j|i)log2(p_(j|i)))
tol=.01;
sig_range=[0 0.7*mean(range(dat))];%starting range of sigma values to search
p_i=zeros(1,N);
sigma=zeros(N,1);%list of sigmas used in computing similarities in the higher dim space
perp_test=0;
G=@(d,sig) exp(-d.^2/(2*sig^2));

% find ideal sigma for each high-Dim point using a binary search over sigma.
% At each step we compute the perplexity which increases monotonically over sigma.
% The procedure is to fix a value of entropy (H) for all points, and find the sigma
% which produces that entropy. Larger sigma produces larger entropy.

for i=1:N
    i
    sig_range=[0 0.7*mean(range(dat))];
    sig_test=(sig_range(1)+sig_range(2))/2;
    c=1;
    perp_test=0;
    while abs(perp_targ-perp_test)>tol
        for j=1:N
            p_i(j)=G(norm(dat(i,:)-dat(j,:)), sig_test);%compute similarity of point i with all others
        end
        p_i(i)=0;%p_ii=0
        p_i=p_i/sum(p_i,2);%make similarity values with all points into a probability distribution, p(i|j)
        p_i(p_i==0)=[];%must remove the zero entry to prevent -Inf in H
        p_i(isnan(p_i))=[];
        H=-sum(p_i.*log2(p_i)/log2(N),2);%NORMALIZED entropy, a measure of the heterogeneity/impurity of the probability distr. eg if each event has equal prob, entropy is maximized
        perp_test=2^H;%perplexity: THIS CANT BE GREATER THAN 2 BECAUSE MAX ENTROPY IS 1
        
%         figure(1)
%         plot(c,sig_test,'o')
%         hold on
        c=c+1;
%         pause(0.2)
        sig_test_last=sig_test;
        if perp_test>perp_targ%do binary search
            sig_range=[sig_range(1) sig_test];
            sig_test=(sig_range(1)+sig_range(2))/2;
        elseif perp_test<perp_targ
            sig_range=[sig_test sig_range(2)];
            sig_test=(sig_range(1)+sig_range(2))/2;
        end
        
        if abs(sig_test_last-sig_test)<=0.001
            disp('could not get within tol of perp_targ')
            break
        end
    end
    sigma(i)=sig_test;
%     perp_test;
%     sig_test;
    
end